#!/usr/bin/env bash
# Train a bevdepth/ config on PCCR R1 (2 GPUs per model).
#
#   bash tools/bevdepth_train.sh CONFIG GPU_IDS [PORT] [extra args for tools/train.py]
#   bash tools/bevdepth_train.sh configs/gen2e/bevdepth-r50-pccr.py 0,1 29510
#
# CONFIG is relative to bevdepth/. Logs and checkpoints go to
# bevdepth/work_dirs/<config name>/R1 (override with WORK_DIR=..., relative to bevdepth/).
# Run it in the gen2e-bevdepth env. tools/dist_train.sh already passes --seed 0.
set -euo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,10p' "$0"
    exit 1
fi
CONFIG=$1
GPU_IDS=$2
PORT=${3:-29510}
shift $(( $# >= 3 ? 3 : 2 ))

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO/bevdepth"
NAME=$(basename "$CONFIG" .py)
NGPU=$(awk -F, '{print NF}' <<< "$GPU_IDS")
WORK_DIR=${WORK_DIR:-work_dirs/$NAME/R1}
mkdir -p "$WORK_DIR"

echo "[gen2e] train $CONFIG on GPUs $GPU_IDS (port $PORT) -> bevdepth/$WORK_DIR"
CUDA_VISIBLE_DEVICES=$GPU_IDS PORT=$PORT \
    bash tools/dist_train.sh "$CONFIG" "$NGPU" --work-dir "$WORK_DIR" "$@"
