#!/usr/bin/env bash
# ImageNet initialisation for BEVFormer-S's R101-DCN backbone (Gen2E decision 2026-11-22):
# detectron2 caffe ResNet-101, mmcv model zoo 'open-mmlab://detectron2/resnet101_caffe'. FCOS3D
# (the official small's load_from) was trained on nuScenes starting from these weights.
#
#   bash tools/download_bevformer_pretrained.sh
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
DST="$REPO/bevformer/ckpts/resnet101_msra-6cc46731.pth"
URL=https://download.openmmlab.com/pretrain/third_party/resnet101_msra-6cc46731.pth
SHA256=6cc46731a982b9d992f81c62e0220fd524273f5464967a2bd82603e85619f314
mkdir -p "$(dirname "$DST")"
[ -f "$DST" ] || curl -sSfL -o "$DST" "$URL"
echo "$SHA256  $DST" | sha256sum -c -
