# -*- coding: utf-8 -*-
import math
import torch
import torch.nn.functional as F
from unicore.losses import register_loss
from . import lorentz as L
from .three_hybrid_loss import ThreeHybridLoss


@register_loss("mixed_modality_loss")
class MixedModalityLoss(ThreeHybridLoss):
    """Mixed-modality pretraining loss.

    Complete assays (seq + pocket + ligand) train the full loss:
      HCC_prot + HCC_poc + CHL + R_ang/R_het
    Missing-pocket assays (seq + ligand only) train only:
      HCC_prot
    One batch can mix both kinds.
    """

    def forward(self, model, sample, reduce=True, fix_encoder=False):
        if self.training:
            h_prot, h_poc, h_mol, aux = model(
                **sample["lig"]["net_input"],
                protein_sequences=sample["protein"],
                features_only=True,
                fix_encoder=fix_encoder,
                is_train=self.training,
                return_aux=True,
            )
        else:
            h_prot, h_poc, h_mol = model(
                **sample["lig"]["net_input"],
                protein_sequences=sample["protein"],
                features_only=True,
                fix_encoder=fix_encoder,
                is_train=self.training,
            )
        B = h_mol.size(0)
        kappa = model.curv.exp().detach()
        logit_scale = model.logit_scale.exp().detach()

        # ---- shared terms on ALL assays ----
        loss_dict_prot = self.compute_hcc_pair_official_style(
            emb_poc=h_prot,
            emb_mol=h_mol,
            batch_list=sample["batch_list"],
            act_list=sample["act_list"],
            uniprot_poc=sample.get("uniprot_list"),
            uniprot_mol=sample.get("lig_uniprot_list"),
            pocket_lig_smiles=sample.get("pocket_lig_smiles"),
            lig_smiles=sample["lig"]["smi_name"],
            logit_scale=logit_scale,
        )
        total_loss = self.alpha_prot * loss_dict_prot["loss"]

        # ---- pocket-dependent terms (complete assays only) ----
        loss_cone = torch.zeros(1, device=h_mol.device)
        R_ang = torch.zeros(1, device=h_mol.device)
        R_het = torch.zeros(1, device=h_mol.device)
        loss_hcc_poc = torch.zeros(1, device=h_mol.device)
        loss_poc_pocket = torch.zeros(1, device=h_mol.device)
        loss_poc_mol = torch.zeros(1, device=h_mol.device)
        loss_poc_rank = torch.zeros(1, device=h_mol.device)
        loss_gen = torch.zeros(1, device=h_mol.device)

        if sample.get("pocket") is not None:
            poc_idx = sample.get("pocket_assay_idx", [])
            if self.training:
                h_poc, u_poc = model.pocket_forward(
                    return_aux=True, **sample["pocket"]["net_input"]
                )
            else:
                h_poc = model.pocket_forward(**sample["pocket"]["net_input"])

            # molecule subset belonging to complete assays, and re-mapped spans
            keep_set = set()
            batch_list_full, act_list_full, uniprot_full = [], [], []
            offset = 0
            for i in poc_idx:
                s, e = sample["batch_list"][i]
                keep_set.update(range(s, e))
                batch_list_full.append([offset, offset + (e - s)])
                offset += (e - s)
                act_list_full.append(sample["act_list"][i])
                uniprot_full.append(sample["uniprot_list"][i])
            mol_keep = sorted(keep_set)
            h_mol_full = h_mol[mol_keep]
            lig_smiles_full = [smi for i, smi in enumerate(sample["lig"]["smi_name"]) if i in keep_set]

            # pocket HCC on complete subset
            loss_dict_poc = self.compute_hcc_pair_official_style(
                emb_poc=h_poc,
                emb_mol=h_mol_full,
                batch_list=batch_list_full,
                act_list=act_list_full,
                uniprot_poc=uniprot_full,
                uniprot_mol=None,
                pocket_lig_smiles=None,
                lig_smiles=lig_smiles_full,
                logit_scale=logit_scale,
            )
            loss_hcc_poc = loss_dict_poc["loss"]
            loss_poc_pocket = loss_dict_poc["loss_pocket"]
            loss_poc_mol = loss_dict_poc["loss_mol"]
            loss_poc_rank = loss_dict_poc["loss_rank"]
            total_loss = total_loss + self.alpha_poc * loss_hcc_poc

            # cone / ranking on complete subset
            poc_space = h_poc[:, 1:]
            lig_space = h_mol_full[:, 1:]
            poc_idx_flat = []
            for i, (s, e) in enumerate(batch_list_full):
                poc_idx_flat += [i] * (e - s)
            poc_idx_flat = torch.tensor(poc_idx_flat, device=h_poc.device)
            poc_sel = poc_space[poc_idx_flat]
            dist_mat = L.pairwise_dist(poc_sel, lig_space, curv=kappa)
            dist = dist_mat.diagonal()
            device = dist.device
            phi = L.oxy_angle(lig_space, poc_space[poc_idx_flat], curv=kappa)
            omega = L.half_aperture(poc_space[poc_idx_flat], curv=kappa)
            act_flat = torch.tensor([x for sub in act_list_full for x in sub],
                                    device=h_poc.device, dtype=torch.float32)
            bounds = self.bounds.to(h_poc.device)
            bucket = torch.bucketize(act_flat, bounds)
            r_k = self.chl_r0 + bucket.float() * self.chl_dr
            eta_k = self.chl_eta0 - bucket.float() * self.chl_deta
            Nl = dist.size(0)
            if Nl > 0:
                L_rad = F.relu(dist - r_k).sum() / math.sqrt(Nl)
                L_ang = F.relu(phi - eta_k * omega).sum() / math.sqrt(Nl)
                loss_cone = self.lambda_rad * L_rad + self.lambda_ang * L_ang
                m_margin = 0.15
                R_ang = F.relu(phi - eta_k * omega + m_margin).sum() / math.sqrt(Nl)
            beta = 80.5
            cnt_het = 0
            offset = 0
            for i_poc, (s, e) in enumerate(batch_list_full):
                L_i = e - s
                if L_i < 1:
                    continue
                d_i = dist[offset:offset + L_i].detach()
                rank = (d_i.unsqueeze(0) < d_i.unsqueeze(1)).float().sum(1) + 1
                w = torch.exp(-beta * (rank - 1) / L_i)
                logits_row = torch.matmul(poc_space[i_poc:i_poc + 1], lig_space.T) * logit_scale
                row_probs = F.softmax(logits_row[0, s:e], dim=-1)
                pos_mask = act_flat[offset:offset + L_i] < 5
                if pos_mask.any():
                    R_het = R_het + (-(w[pos_mask] * row_probs[pos_mask].log()).sum() / (w[pos_mask].sum() + 1e-9))
                    cnt_het += 1
                offset += L_i
            R_het = R_het / max(cnt_het, 1)
            loss_reg = self.lambda_het * R_het + self.lambda_angu * R_ang
            total_loss = total_loss + self.gamma_chl * loss_cone + loss_reg

            # seq -> pocket generation loss on the *complete* subset.
            # u_prot (from the joint forward) covers every assay; the complete
            # ones are at positions poc_idx, in the same ascending order as the
            # pocket batch returned by pocket_forward -> 1:1 pairing holds.
            if self.training and self.gen_space == "tangent":
                idx = torch.tensor(poc_idx, device=u_poc.device, dtype=torch.long)
                u_seq = aux["u_prot"][idx]              # [len(poc_idx), 128]
                loss_gen = self.compute_gen_loss(model, u_seq, u_poc)
                total_loss = total_loss + self.lambda_gen * loss_gen

        if self.training:
            logging_output = {
                "loss":             total_loss.data,
                "sample_size":      B,
                "loss_cone":        loss_cone.item(),
                "R_ang":            R_ang.item(),
                "R_het":            R_het.item(),
                "loss_gen":         loss_gen.item(),
                "loss_hcc_prot":    loss_dict_prot["loss"].item(),
                "loss_prot_pocket": loss_dict_prot["loss_pocket"].item(),
                "loss_prot_mol":    loss_dict_prot["loss_mol"].item(),
                "loss_prot_rank":   loss_dict_prot["loss_rank"].item(),
                "loss_hcc_poc":     loss_hcc_poc.item(),
                "loss_poc_pocket":  loss_poc_pocket.item(),
                "loss_poc_mol":     loss_poc_mol.item(),
                "loss_poc_rank":    loss_poc_rank.item(),
            }
        else:
            sim_masked = loss_dict_prot["sim_masked"].detach()
            sample_size = B
            targets = torch.arange(sample_size, dtype=torch.long, device=sim_masked.device)
            probs = F.softmax(sim_masked[:, :sample_size].float(), dim=-1)
            logging_output = {
                "loss":        torch.tensor(0., device=sim_masked.device),
                "prob":        probs.data,
                "target":      targets,
                "smi_name":    sample["lig"]["smi_name"],
                "sample_size": sample_size,
            }

        return total_loss, B, logging_output
