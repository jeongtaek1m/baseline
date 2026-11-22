#!/usr/bin/env bash
# Train a bevformer/ config on PCCR R1 (2 GPUs per model).
#
#   bash tools/bevformer_train.sh CONFIG GPU_IDS [PORT] [extra args for tools/train.py]
#   bash tools/bevformer_train.sh projects/configs/gen2e/bevformer_small_pccr.py 2,3 29520
#
# CONFIG is relative to bevformer/. Logs and checkpoints go to
# bevformer/work_dirs/<config name>/R1 (override with WORK_DIR=..., relative to bevformer/).
# Run it in the gen2e-bevformer env. tools/train.py seeds with 0 by default and
# tools/dist_train.sh passes --deterministic (both upstream).
set -euo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,11p' "$0"
    exit 1
fi
CONFIG=$1
GPU_IDS=$2
PORT=${3:-29520}
shift $(( $# >= 3 ? 3 : 2 ))

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO/bevformer"
NAME=$(basename "$CONFIG" .py)
NGPU=$(awk -F, '{print NF}' <<< "$GPU_IDS")
WORK_DIR=${WORK_DIR:-work_dirs/$NAME/R1}
mkdir -p "$WORK_DIR"

echo "[gen2e] train $CONFIG on GPUs $GPU_IDS (port $PORT) -> bevformer/$WORK_DIR"
CUDA_VISIBLE_DEVICES=$GPU_IDS PORT=$PORT \
    bash tools/dist_train.sh "$CONFIG" "$NGPU" --work-dir "$WORK_DIR" "$@"
