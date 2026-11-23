#!/usr/bin/env bash
# Score a checkpoint of a co-author-base Megvii PCCR exp on the 14 PCCR test rigs with the co-author's evaluation
# (PCCRDetEvaluator -> megvii_bevdepth/pccr/eval_pccr_devkit.py, nuscenes-devkit 1.1.10), fp32, 1 GPU.
#
#   bash tools/megvii_pccr_test_rigs.sh EXP GPU [CKPT]
#     EXP: bev_depth_lss_r50_pccr_24e (plain BEVDepth) | pdbev_lss_r50_pccr_24e_A | pdbev_lss_r50_pccr_24e_B
#     CKPT default: the newest epoch=23 checkpoint under RUN (default megvii_bevdepth/outputs/<EXP>)
#
# Results: <RUN>/pccr_test/<ckpt name>/<RIG>/{results_nusc.json, eval.json} (+ checkpoint.txt); rigs already scored
# are skipped; RIGS="R1 R2" scores only those (e.g. half the rigs per GPU). Prints the co-author's summary table.
set -euo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,11p' "$0"
    exit 1
fi
EXP=$1
GPU=$2
REPO=$(cd "$(dirname "$0")/.." && pwd)
RUN=${RUN:-$REPO/megvii_bevdepth/outputs/$EXP}
CKPT=${3:-$(ls -t "$RUN"/lightning_logs/version_*/checkpoints/epoch=23-*.ckpt 2>/dev/null | head -1)}
[ -f "$CKPT" ] || { echo "no checkpoint ($CKPT)"; exit 1; }
CKPT=$(realpath "$CKPT")
read -r -a RIGS <<< "${RIGS:-R1 R1-t R1-r R1-f R1-c6 R1-c10 R2 R3 R4 R5 R6 R7 R8 R9}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
# fp32 inference (as every PCCR row): in fp16 the camera-aware MLP input overflows on rigs whose intrinsics differ
export PCCR_PRECISION=32
export PCCR_DEVKIT_PY=${PCCR_DEVKIT_PY:-$HOME/miniconda3/envs/gen2e-megvii/bin/python}
PY=${PY:-$HOME/miniconda3/envs/gen2e-megvii/bin/python}
OUTD=$RUN/pccr_test/$(basename "${CKPT%.ckpt}")
mkdir -p "$OUTD"
echo "$CKPT" > "$OUTD/checkpoint.txt"
cd "$REPO/megvii_bevdepth"
echo "[gen2e] $EXP: scoring $CKPT on ${#RIGS[@]} rigs (${RIGS[*]}) -> $OUTD"
CUDA_VISIBLE_DEVICES=$GPU MASTER_PORT=${MASTER_PORT:-$((29600 + GPU))} "$PY" bevdepth/exps/pccr/$EXP.py \
    --pccr_root "$REPO/data/pccr" --ckpt_path "$CKPT" -e -b 4 --gpus 1 --precision 32 \
    --test_rigs "${RIGS[@]}" --eval_out "$OUTD" --default_root_dir "$RUN"
"$PY" pccr/summarize_pccr.py "$OUTD"
