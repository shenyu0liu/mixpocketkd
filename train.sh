#!/usr/bin/env bash
# MixPocketKD training on a mixed corpus of complete and pocket-missing assays.
#
# Usage:
#   bash train.sh [LAMBDA_GEN] [SAVE_NAME]
#
#   LAMBDA_GEN   weight of the pocket-embedding generator loss (default: 0.1).
#                Set to 0 to train without the generator.
#   SAVE_NAME    run directory under $SAVE_ROOT. Default:
#                screen_pocket_kd_gen_lg<LAMBDA_GEN>, screen_pocket_kd_gen
#                for LAMBDA_GEN=0.1, or screen_pocket_kd when LAMBDA_GEN is 0.
#
# Environment overrides:
#   PYTHON, DATA_DIR, SAVE_ROOT, PRETRAIN_DIR, CUDA_VISIBLE_DEVICES,
#   MAX_EPOCH, BATCH_SIZE, LR, SEED, VALID_SET, ALLOW_RANDOM_INIT
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash train.sh [LAMBDA_GEN] [SAVE_NAME]

  LAMBDA_GEN   weight of the pocket-embedding generator loss (default: 0.1);
               0 trains without the generator.
  SAVE_NAME    run directory under $SAVE_ROOT (default: derived from LAMBDA_GEN).

Environment: PYTHON, DATA_DIR, SAVE_ROOT, PRETRAIN_DIR, VALID_SET, ALLOW_RANDOM_INIT,
             CUDA_VISIBLE_DEVICES, MAX_EPOCH, BATCH_SIZE, LR, SEED
EOF
    exit "${1:-0}"
}

case "${1:-}" in
    -h|--help) usage 0 ;;
esac

LAMBDA_GEN="${1:-0.1}"
if ! [[ "${LAMBDA_GEN}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "error: LAMBDA_GEN must be a non-negative number, got '${LAMBDA_GEN}'" >&2
    exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
SAVE_ROOT="${SAVE_ROOT:-${PROJECT_ROOT}/save_root}"
PRETRAIN_DIR="${PRETRAIN_DIR:-${PROJECT_ROOT}/pretrain}"
PYTHON="${PYTHON:-python}"
MAX_EPOCH="${MAX_EPOCH:-50}"
BATCH_SIZE="${BATCH_SIZE:-24}"
LR="${LR:-1e-4}"
SEED="${SEED:-1}"
VALID_SET="${VALID_SET:-CASF}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export WORLD_SIZE="${WORLD_SIZE:-1}"
export RANK="${RANK:-0}"
export LOCAL_RANK="${LOCAL_RANK:-0}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

DATA_FILES=(
    train_lig_all_blend.lmdb
    train_prot_all_blend.lmdb
    train_label_pdbbind_seq.json
    train_label_blend_orig.json
    train_label_blend_seq_full.json
    fep_repeat_ligands_can.json
)
missing_files=()
for f in "${DATA_FILES[@]}"; do
    if [ ! -e "${DATA_DIR}/${f}" ]; then
        missing_files+=("${f}")
    fi
done
if [ "${#missing_files[@]}" -gt 0 ]; then
    echo "error: missing training files under ${DATA_DIR}: ${missing_files[*]}" >&2
    exit 2
fi
case "${VALID_SET}" in
    CASF) VALID_FILES=(valid_lig.lmdb valid_prot.lmdb valid_label_seq.json) ;;
    FEP)  VALID_FILES=(FEP/ligands.lmdb FEP/proteins.lmdb FEP/fep_labels.json) ;;
    *)    VALID_FILES=() ;;
esac
missing_valid=()
for f in "${VALID_FILES[@]}"; do
    if [ ! -e "${DATA_DIR}/${f}" ]; then
        missing_valid+=("${f}")
    fi
done
if [ "${#missing_valid[@]}" -gt 0 ]; then
    echo "error: missing ${VALID_SET} validation files under ${DATA_DIR}: ${missing_valid[*]}" >&2
    echo "       set VALID_SET to an available validation split" >&2
    exit 2
fi

if [ "${VALID_SET}" = "CASF" ]; then
    VALIDATION_ARGS=(--best-checkpoint-metric valid_bedroc --maximize-best-checkpoint-metric)
    CKPT_NAME="checkpoint_best.pt"
else
    VALIDATION_ARGS=(--disable-validation)
    CKPT_NAME="checkpoint_last.pt"
fi

if awk "BEGIN{exit !(${LAMBDA_GEN} == 0)}"; then
    DEFAULT_NAME="screen_pocket_kd"
elif awk "BEGIN{exit !(${LAMBDA_GEN} == 0.1)}"; then
    DEFAULT_NAME="screen_pocket_kd_gen"
else
    DEFAULT_NAME="screen_pocket_kd_gen_lg${LAMBDA_GEN}"
fi
SAVE_NAME="${2:-${DEFAULT_NAME}}"

SAVE_DIR="${SAVE_ROOT}/${SAVE_NAME}/savedir_screen"
TMP_SAVE_DIR="${SAVE_ROOT}/${SAVE_NAME}/tmp_save_dir_screen"
TSB_DIR="${SAVE_ROOT}/${SAVE_NAME}/tsb_dir_screen"
LOG="${SAVE_ROOT}/${SAVE_NAME}/train.log"
mkdir -p "${SAVE_DIR}" "${TMP_SAVE_DIR}" "${TSB_DIR}"

PRETRAIN_ARGS=()
missing_pretrain=()
if [ -f "${PRETRAIN_DIR}/mol_pre_no_h_220816.pt" ]; then
    PRETRAIN_ARGS+=(--finetune-mol-model "${PRETRAIN_DIR}/mol_pre_no_h_220816.pt")
else
    missing_pretrain+=(mol_pre_no_h_220816.pt)
fi
if [ -f "${PRETRAIN_DIR}/pocket_pre_220816.pt" ]; then
    PRETRAIN_ARGS+=(--finetune-pocket-model "${PRETRAIN_DIR}/pocket_pre_220816.pt")
else
    missing_pretrain+=(pocket_pre_220816.pt)
fi
if [ "${#missing_pretrain[@]}" -gt 0 ] && [ "${ALLOW_RANDOM_INIT:-0}" != "1" ]; then
    echo "error: pretrained Uni-Mol weights not found in ${PRETRAIN_DIR}: ${missing_pretrain[*]}" >&2
    echo "       place them there, set PRETRAIN_DIR, or set ALLOW_RANDOM_INIT=1" >&2
    exit 2
fi

echo "lambda_gen=${LAMBDA_GEN}  save_dir=${SAVE_DIR}  log=${LOG}"

"${PYTHON}" -m unicore_cli.train "${DATA_DIR}" \
    --user-dir "${PROJECT_ROOT}/unimol" \
    --task train_task \
    --arch three_hybrid_model \
    --loss mixed_modality_loss \
    --mixed-modality \
    --train-subset train \
    --valid-subset valid \
    --valid-set "${VALID_SET}" \
    --num-workers 0 \
    --ddp-backend c10d \
    --max-pocket-atoms 256 \
    --optimizer adam \
    --adam-betas "(0.9, 0.999)" \
    --adam-eps 1e-8 \
    --clip-norm 1.0 \
    --lr-scheduler polynomial_decay \
    --lr "${LR}" \
    --warmup-ratio 0.06 \
    --max-epoch "${MAX_EPOCH}" \
    --batch-size "${BATCH_SIZE}" \
    --batch-size-valid 32 \
    --fp16 \
    --fp16-init-scale 4 \
    --fp16-scale-window 256 \
    --update-freq 1 \
    --seed "${SEED}" \
    --tensorboard-logdir "${TSB_DIR}" \
    --log-interval 100 \
    --log-format simple \
    --validate-interval 1 \
    --all-gather-list-size 2048000 \
    --save-dir "${SAVE_DIR}" \
    --tmp-save-dir "${TMP_SAVE_DIR}" \
    --keep-best-checkpoints 8 \
    --keep-last-epochs 10 \
    --find-unused-parameters \
    "${VALIDATION_ARGS[@]}" \
    ${PRETRAIN_ARGS[@]+"${PRETRAIN_ARGS[@]}"} \
    --max-lignum 16 \
    --learn-curv \
    --protein-similarity-thres 1.0 \
    --lambda-gen "${LAMBDA_GEN}" \
    --gen-space tangent \
    >> "${LOG}" 2>&1

echo "log: ${LOG}"
echo "checkpoint: ${SAVE_DIR}/${CKPT_NAME}"
