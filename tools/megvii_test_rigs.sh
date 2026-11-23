#!/usr/bin/env bash
# Predict with a Megvii BEVDepth PCCR checkpoint on the test split of every PCCR rig, score each
# rig with PCCR's evaluation (tools/score_megvii_pccr.py) and print the 7-class mAP row.
#
#   bash tools/megvii_test_rigs.sh CKPT GPU_IDS [RIG ...]
#
# Results go to <run dir>/test_on_<RIG>_test/ (results_nusc.json, metrics_summary.json,
# checkpoint.txt, data.txt), where <run dir> is the folder holding lightning_logs/
# (override with WORK=...). Runs the Megvii code in gen2e-megvii and the scorer in gen2e-bevdepth.
set -euo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,10p' "$0"
    exit 1
fi
CKPT=$(realpath "$1")
GPU_IDS=$2
shift 2
RIGS=("$@")
if [ ${#RIGS[@]} -eq 0 ]; then
    RIGS=(R1 R1-c6 R1-c10 R1-f R1-r R1-t R2 R3 R4 R5 R6 R7 R8 R9)
fi
REPO=$(cd "$(dirname "$0")/.." && pwd)
# one BLAS/OpenMP thread per process: the 8 data workers otherwise oversubscribe the CPUs
# (per-sample load 0.15 s single-threaded, slower with 8 threads); no effect on results
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
PY=${PY:-$HOME/miniconda3/envs/gen2e-megvii/bin/python}
SCORE_PY=${SCORE_PY:-$HOME/miniconda3/envs/gen2e-bevdepth/bin/python}
NGPU=$(awk -F, '{print NF}' <<< "$GPU_IDS")
WORK=${WORK:-$(echo "$CKPT" | sed 's#/lightning_logs/.*##')}
SPLIT=${SPLIT:-test}

cd "$REPO/megvii_bevdepth"
for RIG in "${RIGS[@]}"; do
    OUTD="$WORK/test_on_${RIG}_${SPLIT}"
    mkdir -p "$OUTD"
    echo "[gen2e] $RIG: $(basename "$CKPT") -> $OUTD (infos ${RIG}_megvii_infos_${SPLIT}.pkl)"
    PCCR_RIG=$RIG PCCR_SPLIT=$SPLIT PCCR_OUT=$OUTD CUDA_VISIBLE_DEVICES=$GPU_IDS \
        "$PY" bevdepth/exps/pccr/bev_depth_lss_r50_256x704_128x128_24e_pccr.py \
        -p --ckpt_path "$CKPT" --gpus "$NGPU" -b 4 --precision 32 --default_root_dir "$WORK" \
        > "$OUTD.log" 2>&1
    echo "$CKPT" > "$OUTD/checkpoint.txt"
    printf 'data_root=%s\ninfos=%s\n' "$(realpath "$REPO/data/pccr/$RIG")" \
        "$(realpath "data/pccr_infos/${RIG}_megvii_infos_${SPLIT}.pkl")" > "$OUTD/data.txt"
    CUDA_VISIBLE_DEVICES= "$SCORE_PY" -I "$REPO/tools/score_megvii_pccr.py" "$OUTD/results_nusc.json" \
        "$RIG" --split "$SPLIT" 2>&1 | tail -1 | tee -a "$OUTD.log"
done

if [ "$SPLIT" = test ]; then
    python3 "$REPO/tools/pccr_metric.py" summarize "$WORK/test_on_*_test/**/metrics_summary.json"
fi
