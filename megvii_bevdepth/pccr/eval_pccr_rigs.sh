#!/usr/bin/env bash
# PCCR port: new file. Score a trained PCCR BEVDepth checkpoint on the 14 PCCR test rigs (1 GPU), then print the table.
#   GPU=2 nohup bash pccr/eval_pccr_rigs.sh plain [ckpt] > logs/eval_plain.log 2>&1 &
#   RIGS="R1 R2" scores only those rigs (e.g. half per GPU); rigs already scored are skipped; extra exp arguments
#   (e.g. --final_h 384 for a 384-trained model) via EXTRA="...". Default ckpt: the newest epoch=23 checkpoint.
# Results: outputs/<exp>/pccr_test/<ckpt name>/<rig>/{results_nusc.json, eval.json}, scored by pccr/eval_pccr_devkit.py
# in $PCCR_DEVKIT_PY (nuscenes-devkit 1.1.10, the python every PCCR row is scored with on Poseidon).
set -eo pipefail
VARIANT=${1:?usage: GPU=2 eval_pccr_rigs.sh <plain|unidrive|unidrive_r1|canon|canon_vs> [ckpt]}
case $VARIANT in
  plain) EXP=bev_depth_lss_r50_pccr_24e ;;
  unidrive) EXP=bev_depth_lss_r50_pccr_24e_unidrive ;;
  unidrive_r1) EXP=bev_depth_lss_r50_pccr_24e_unidrive_r1 ;;
  canon) EXP=bev_depth_lss_r50_pccr_24e_canon ;;
  canon_vs) EXP=bev_depth_lss_r50_pccr_24e_canon_vs ;;
  *) echo "unknown variant $VARIANT"; exit 1 ;;
esac
cd "$(dirname "$0")/.."
CKPT=${2:-$(ls -t outputs/$EXP/lightning_logs/version_*/checkpoints/epoch=23-*.ckpt 2>/dev/null | head -1)}
[ -f "$CKPT" ] || { echo "no checkpoint ($CKPT)"; exit 1; }
read -r -a RIGS <<< "${RIGS:-R1 R1-t R1-r R1-f R1-c6 R1-c10 R2 R3 R4 R5 R6 R7 R8 R9}"
ENV=${BEVDEPTH_ENV:-/home/giyong/miniforge3/envs/bevdepth_megvii}
export PATH=$ENV/bin:$PATH CUDA_VISIBLE_DEVICES=${GPU:?set GPU} MASTER_PORT=${PORT:-$((29600 + GPU))} OMP_NUM_THREADS=2
# PCCR port: inference in fp32 (training keeps the official amp). In fp16 the camera-aware MLP input of BEVDepth
# (BatchNorm1d of camera parameters that are constant on R1, |z| up to 2e5 on other rigs) overflows to inf / NaN and
# no box survives (R6: 0.0 in fp16, 7.4 mAP7 in fp32); every other PCCR row is scored in fp32.
export PCCR_PRECISION=${PCCR_PRECISION:-32}
export PCCR_DEVKIT_PY=${PCCR_DEVKIT_PY:-/data/giyong/envs/bevformer_canon/bin/python}
OUT=outputs/$EXP/pccr_test/$(basename "${CKPT%.ckpt}")
echo "[$(TZ=Asia/Seoul date '+%F %H:%M') KST] scoring $CKPT on ${#RIGS[@]} rigs (${RIGS[*]}) -> $OUT"
python bevdepth/exps/pccr/$EXP.py --ckpt_path "$CKPT" -e -b 4 --gpus 1 --test_rigs "${RIGS[@]}" --eval_out "$OUT" $EXTRA
echo "[$(TZ=Asia/Seoul date '+%F %H:%M') KST] done"
python pccr/summarize_pccr.py "$OUT"
