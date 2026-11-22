#!/usr/bin/env bash
# Evaluate a model trained by tools/bevformer_train.sh on the test split of every PCCR rig,
# then print the 7-class mAP row.
#
#   bash tools/bevformer_test_rigs.sh WORK_DIR GPU_IDS [PORT] [RIG ...]
#   bash tools/bevformer_test_rigs.sh bevformer/work_dirs/bevformer_small_pccr/R1 2,3
#
# Uses the config copy that mmdet saved in WORK_DIR and WORK_DIR/latest.pth (as PCCR's
# sequential_test.sh does). Results go to WORK_DIR/test_on_<RIG>_test/. Run it in the
# gen2e-bevformer env. bevformer/tools/dist_test.sh already passes --eval bbox.
set -euo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,10p' "$0"
    exit 1
fi
WORK_DIR=$(realpath "$1")
GPU_IDS=$2
PORT=${3:-29540}
shift $(( $# >= 3 ? 3 : 2 ))
RIGS=("$@")
if [ ${#RIGS[@]} -eq 0 ]; then
    RIGS=(R1 R1-c6 R1-c10 R1-f R1-r R1-t R2 R3 R4 R5 R6 R7 R8 R9)
fi

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO/bevformer"
NGPU=$(awk -F, '{print NF}' <<< "$GPU_IDS")
CKPT=${CKPT:-$WORK_DIR/latest.pth}
CONFIGS=("$WORK_DIR"/*.py)
if [ ${#CONFIGS[@]} -ne 1 ]; then
    echo "expected exactly one config in $WORK_DIR, found ${#CONFIGS[@]}" >&2
    exit 1
fi
CONFIG=${CONFIGS[0]}

for RIG in "${RIGS[@]}"; do
    ROOT="../data/pccr/$RIG/"
    OUT="$WORK_DIR/test_on_${RIG}_test"
    echo "[gen2e] $RIG: $(basename "$CKPT") -> $OUT"
    CUDA_VISIBLE_DEVICES=$GPU_IDS PORT=$PORT \
        bash tools/dist_test.sh "$CONFIG" "$CKPT" "$NGPU" \
        --cfg-options data.test.data_root="$ROOT" data.test.ann_file="${ROOT}${RIG}_infos_test.pkl" \
        --eval-options jsonfile_prefix="$OUT" 2>&1 | tee "$OUT.log"
done

python3 "$REPO/tools/pccr_metric.py" summarize "$WORK_DIR/test_on_*_test/**/metrics_summary.json"
