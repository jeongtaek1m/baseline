#!/usr/bin/env bash
# Evaluate a model trained by tools/bevdepth_train.sh on the test split of every PCCR rig,
# then print the 7-class mAP row.
#
#   bash tools/bevdepth_test_rigs.sh WORK_DIR GPU_IDS [PORT] [RIG ...]
#   bash tools/bevdepth_test_rigs.sh bevdepth/work_dirs/bevdepth-r50-pccr/R1 0,1
#
# Uses the config copy that mmdet saved in WORK_DIR and WORK_DIR/latest.pth (as PCCR's
# sequential_test.sh does). Results go to WORK_DIR/test_on_<RIG>_test/.
set -euo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,9p' "$0"
    exit 1
fi
WORK_DIR=$(realpath "$1")
GPU_IDS=$2
shift 2
PORT=29530
if [[ ${1:-} =~ ^[0-9]+$ ]]; then  # optional PORT, then rig names
    PORT=$1
    shift
fi
RIGS=("$@")
if [ ${#RIGS[@]} -eq 0 ]; then
    RIGS=(R1 R1-c6 R1-c10 R1-f R1-r R1-t R2 R3 R4 R5 R6 R7 R8 R9)
fi

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO/bevdepth"
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
    echo "[gen2e] $RIG: $(basename "$CKPT") -> $OUT (data_root=$ROOT ann_file=${ROOT}${RIG}_infos_test.pkl)"
    CUDA_VISIBLE_DEVICES=$GPU_IDS PORT=$PORT \
        bash tools/dist_test.sh "$CONFIG" "$CKPT" "$NGPU" --eval bbox \
        --cfg-options data.test.data_root="$ROOT" data.test.ann_file="${ROOT}${RIG}_infos_test.pkl" \
        --eval-options jsonfile_prefix="$OUT" 2>&1 | tee "$OUT.log"
    realpath "$CKPT" > "$OUT/checkpoint.txt"  # read by pccr_metric.py summarize
    # the data this rig was evaluated on (physical paths; the test log does not echo cfg-options)
    printf 'data_root=%s\nann_file=%s\n' "$(realpath "$ROOT")" \
        "$(realpath "${ROOT}${RIG}_infos_test.pkl")" > "$OUT/data.txt"
done

python3 "$REPO/tools/pccr_metric.py" summarize "$WORK_DIR/test_on_*_test/**/metrics_summary.json"
