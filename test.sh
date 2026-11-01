#!/usr/bin/env bash
# MixPocketKD inference on virtual screening benchmarks.
#
# Usage:
#   bash test.sh <task> <arch> <checkpoint> <results_dir> [--seq-only] [extra args]
#
#   task         DUDE | DEKOIS | PCBA | ALL
#   arch         model architecture, e.g. three_hybrid_model
#   checkpoint   path to a trained checkpoint
#   results_dir  output directory
#   --seq-only   score with the protein-sequence branch only
#                (alpha_poc=0, alpha_prot=1); default is alpha_poc=1, alpha_prot=1
#
# Example:
#   bash test.sh ALL three_hybrid_model /path/checkpoint_avg_41-50_vs.pt ./results
#
# Environment overrides:
#   PYTHON, DATA_DIR, CUDA_VISIBLE_DEVICES, BATCH_SIZE, MAX_POCKET_ATOMS
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash test.sh <task> <arch> <checkpoint> <results_dir> [--seq-only] [extra args]

  task         DUDE | DEKOIS | PCBA | ALL
  arch         model architecture, e.g. three_hybrid_model
  checkpoint   path to a trained checkpoint
  results_dir  output directory
  --seq-only   score with the protein-sequence branch only
               (alpha_poc=0, alpha_prot=1); default is alpha_poc=1, alpha_prot=1

Data: $DATA_DIR/{DUD-E, DEKOIS_2.0x, lit_pcba} with dude.json, dekois.json, PCBA.json
Environment: PYTHON, DATA_DIR, CUDA_VISIBLE_DEVICES, BATCH_SIZE, MAX_POCKET_ATOMS
EOF
    exit "${1:-0}"
}

case "${1:-}" in
    -h|--help) usage 0 ;;
esac

if [ "$#" -lt 4 ]; then
    usage 2
fi

TASK="$1"
ARCH="$2"
CKPT="$3"
RESULTS_DIR="$4"
shift 4

case "${TASK}" in
    DUDE|DEKOIS|PCBA|ALL) ;;
    *)
        echo "error: unknown task '${TASK}' (expected DUDE, DEKOIS, PCBA, or ALL)" >&2
        exit 2
        ;;
esac

SEQ_ONLY=0
EXTRA_ARGS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --seq-only) SEQ_ONLY=1; shift ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [ ! -f "${CKPT}" ]; then
    echo "error: checkpoint not found: ${CKPT}" >&2
    exit 2
fi

if [ "${SEQ_ONLY}" -eq 1 ]; then
    ALPHA_POC=0.0
    ALPHA_PROT=1.0
    TAG="seqonly"
else
    ALPHA_POC=1.0
    ALPHA_PROT=1.0
    TAG="full"
fi

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
PYTHON="${PYTHON:-python}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_POCKET_ATOMS="${MAX_POCKET_ATOMS:-511}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export WORLD_SIZE="${WORLD_SIZE:-1}"
export RANK="${RANK:-0}"
export LOCAL_RANK="${LOCAL_RANK:-0}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

case "${TASK}" in
    DUDE)   DATA_ENTRIES=(DUD-E dude.json) ;;
    DEKOIS) DATA_ENTRIES=(DEKOIS_2.0x dekois.json) ;;
    PCBA)   DATA_ENTRIES=(lit_pcba PCBA.json) ;;
    ALL)    DATA_ENTRIES=(DUD-E dude.json DEKOIS_2.0x dekois.json lit_pcba PCBA.json) ;;
esac
missing_entries=()
for entry in "${DATA_ENTRIES[@]}"; do
    if [ ! -e "${DATA_DIR}/${entry}" ]; then
        missing_entries+=("${entry}")
    fi
done
if [ "${#missing_entries[@]}" -gt 0 ]; then
    echo "error: missing benchmark data under ${DATA_DIR}: ${missing_entries[*]}" >&2
    exit 2
fi

mkdir -p "${RESULTS_DIR}"
LOG="${RESULTS_DIR}/test_${TASK}_${TAG}.log"

echo "task=${TASK}  arch=${ARCH}  alpha_poc=${ALPHA_POC}  alpha_prot=${ALPHA_PROT}"
echo "checkpoint=${CKPT}"
echo "results=${RESULTS_DIR}  log=${LOG}"

"${PYTHON}" "${PROJECT_ROOT}/unimol/test.py" "${DATA_DIR}" \
    --user-dir "${PROJECT_ROOT}/unimol" \
    --valid-subset test \
    --results-path "${RESULTS_DIR}" \
    --num-workers 0 \
    --ddp-backend c10d \
    --distributed-world-size 1 \
    --batch-size "${BATCH_SIZE}" \
    --task test_task \
    --loss three_hybrid_loss \
    --arch "${ARCH}" \
    --fp16 \
    --fp16-init-scale 4 \
    --fp16-scale-window 256 \
    --seed 1 \
    --path "${CKPT}" \
    --log-interval 100 \
    --log-format simple \
    --max-pocket-atoms "${MAX_POCKET_ATOMS}" \
    --alpha-poc "${ALPHA_POC}" \
    --alpha-prot "${ALPHA_PROT}" \
    --test-task "${TASK}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    >> "${LOG}" 2>&1

echo "done; log: ${LOG}"
