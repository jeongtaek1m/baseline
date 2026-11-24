#!/usr/bin/env bash
# Train BEVDepth + UDGA (Lp: paper final row, Ld: public config composition) on the co-author's Megvii PCCR base: R1,
# total batch 8 = GPUs x 4, fp32 (decisions 2026-11-23 / 2026-11-24).
#
#   bash tools/udga_megvii_train.sh Lp|Ld GPU_IDS [extra pytorch-lightning / exp args]
#   bash tools/udga_megvii_train.sh Ld 2,3
#
# Checkpoints: megvii_bevdepth/outputs/udga_lss_r50_pccr_24e_<V>/lightning_logs/version_*/checkpoints/ (override
# with OUT=...). Env gen2e-megvii (tools/setup_env_megvii.sh); infos megvii_bevdepth/data/pccr_megvii
# (megvii_bevdepth/scripts/gen_info_pccr.py --pccr-root ../data/pccr).
set -euo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,10p' "$0"
    exit 1
fi
R=$1
GPU_IDS=$2
shift 2
REPO=$(cd "$(dirname "$0")/.." && pwd)
# one BLAS/OpenMP thread per process: the data workers otherwise oversubscribe the CPUs; no effect on results
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export PCCR_PRECISION=32
PY=${PY:-$HOME/miniconda3/envs/gen2e-megvii/bin/python}
NGPU=$(awk -F, '{print NF}' <<< "$GPU_IDS")
EXP=udga_lss_r50_pccr_24e_$R
OUT=${OUT:-$REPO/megvii_bevdepth/outputs/$EXP}
mkdir -p "$OUT"
cd "$REPO/megvii_bevdepth"
echo "[gen2e] UDGA-$R train on GPUs $GPU_IDS (batch ${NGPU}x4, fp32) -> $OUT"
CUDA_VISIBLE_DEVICES=$GPU_IDS MASTER_PORT=${MASTER_PORT:-$((29540 + $(cut -d, -f1 <<< "$GPU_IDS")))} \
    "$PY" bevdepth/exps/pccr/$EXP.py --pccr_root "$REPO/data/pccr" --gpus "$NGPU" -b 4 --precision 32 \
    --default_root_dir "$OUT" "$@"
