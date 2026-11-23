#!/usr/bin/env bash
# Train the official Megvii BEVDepth (PCCR exp) on PCCR R1: total batch 8 = GPUs x 4, FP32.
#
#   bash tools/megvii_train.sh GPU_IDS [extra pytorch-lightning args]
#   bash tools/megvii_train.sh 0,1
#
# Checkpoints: $OUT/lightning_logs/version_*/checkpoints/ (default OUT =
# megvii_bevdepth/outputs/pccr_bevdepth_r50/R1). Runs in the gen2e-megvii env
# (tools/setup_env_megvii.sh). Infos: megvii_bevdepth/scripts/gen_info_pccr.py.
set -euo pipefail

if [ $# -lt 1 ]; then
    sed -n '2,9p' "$0"
    exit 1
fi
GPU_IDS=$1
shift
REPO=$(cd "$(dirname "$0")/.." && pwd)
# one BLAS/OpenMP thread per process: the 8 data workers otherwise oversubscribe the CPUs
# (per-sample load 0.15 s single-threaded, slower with 8 threads); no effect on results
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
PY=${PY:-$HOME/miniconda3/envs/gen2e-megvii/bin/python}
NGPU=$(awk -F, '{print NF}' <<< "$GPU_IDS")
OUT=${OUT:-$REPO/megvii_bevdepth/outputs/pccr_bevdepth_r50/R1}
mkdir -p "$OUT"
cd "$REPO/megvii_bevdepth"
echo "[gen2e] Megvii BEVDepth PCCR train on GPUs $GPU_IDS (batch ${NGPU}x4) -> $OUT"
CUDA_VISIBLE_DEVICES=$GPU_IDS "$PY" bevdepth/exps/pccr/bev_depth_lss_r50_256x704_128x128_24e_pccr.py \
    --gpus "$NGPU" -b 4 --precision 32 --default_root_dir "$OUT" "$@"
