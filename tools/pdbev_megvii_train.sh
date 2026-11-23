#!/usr/bin/env bash
# Train BEVDepth + PD-BEV (perturbation ranges A or B) on the co-author's Megvii PCCR base: R1, total batch 8 =
# GPUs x 4, fp32 (decision 2026-11-23).
#
#   bash tools/pdbev_megvii_train.sh A|B GPU_IDS [extra pytorch-lightning / exp args]
#   bash tools/pdbev_megvii_train.sh A 2,3
#
# Checkpoints: megvii_bevdepth/outputs/pdbev_lss_r50_pccr_24e_<R>/lightning_logs/version_*/checkpoints/ (override
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
EXP=pdbev_lss_r50_pccr_24e_$R
OUT=${OUT:-$REPO/megvii_bevdepth/outputs/$EXP}
mkdir -p "$OUT"
cd "$REPO/megvii_bevdepth"
echo "[gen2e] PD-BEV-$R train on GPUs $GPU_IDS (batch ${NGPU}x4, fp32) -> $OUT"
CUDA_VISIBLE_DEVICES=$GPU_IDS MASTER_PORT=${MASTER_PORT:-$((29540 + $(cut -d, -f1 <<< "$GPU_IDS")))} \
    "$PY" bevdepth/exps/pccr/$EXP.py --pccr_root "$REPO/data/pccr" --gpus "$NGPU" -b 4 --precision 32 \
    --default_root_dir "$OUT" "$@"
