#!/usr/bin/env bash
# PCCR port: new file. Train one PCCR BEVDepth variant on 2 GPUs (official CLI defaults: ddp, amp, clip 5, 24 epochs).
#   GPUS=0,1 PORT=29531 nohup bash pccr/train_pccr.sh plain > logs/train_plain.log 2>&1 &
#   variants: plain | unidrive | unidrive_r1 | canon | canon_vs ; extra arguments go to the exp (e.g. --final_h 384, --lr_official)
# Checkpoints: outputs/<exp>/lightning_logs/version_*/checkpoints/epoch=23-step=*.ckpt (PL keeps the last epoch).
set -eo pipefail
VARIANT=${1:?usage: GPUS=0,1 PORT=29531 train_pccr.sh <plain|unidrive|unidrive_r1|canon|canon_vs> [exp args]}
shift
case $VARIANT in
  plain) EXP=bev_depth_lss_r50_pccr_24e ;;
  unidrive) EXP=bev_depth_lss_r50_pccr_24e_unidrive ;;
  unidrive_r1) EXP=bev_depth_lss_r50_pccr_24e_unidrive_r1 ;;
  canon) EXP=bev_depth_lss_r50_pccr_24e_canon ;;
  canon_vs) EXP=bev_depth_lss_r50_pccr_24e_canon_vs ;;
  *) echo "unknown variant $VARIANT"; exit 1 ;;
esac
GPUS=${GPUS:?set GPUS, e.g. GPUS=0,1}
PORT=${PORT:-29531}
NGPU=$(awk -F, '{print NF}' <<< "$GPUS")
cd "$(dirname "$0")/.."
ENV=${BEVDEPTH_ENV:-/home/giyong/miniforge3/envs/bevdepth_megvii}
export PATH=$ENV/bin:$PATH CUDA_VISIBLE_DEVICES=$GPUS MASTER_PORT=$PORT OMP_NUM_THREADS=2
echo "[$(TZ=Asia/Seoul date '+%F %H:%M') KST] train $EXP on GPUs $GPUS (port $PORT): -b 4 x $NGPU $*"
python bevdepth/exps/pccr/$EXP.py --amp_backend native -b 4 --gpus "$NGPU" "$@"
echo "[$(TZ=Asia/Seoul date '+%F %H:%M') KST] done"
