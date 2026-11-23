#!/usr/bin/env bash
# PCCR port: new file. How the bevdepth_megvii env on Poseidon was made (no conda binary there, so a copy of coin3d).
#   coin3d (torch 1.12.1+cu116, mmcv-full 1.6.0, mmdet 2.28.0) copied with `cp -a` to envs/bevdepth_megvii, then:
#   - shebangs of bin/* pointed at the new prefix;
#   - CoIn3D's develop installs removed from the copy (BEVDet-fork mmdet3d, CoIn3D's nuscenes-devkit, its rasterizer);
#   - official mmdet3d 1.0.0rc4 (PyPI sdist, pure python in rc4), nuscenes-devkit 1.1.10, pytorch-lightning 1.6.0
#     (+ torchmetrics 0.7.3, its contemporary), setuptools 59.5.0 (requirements.txt), tensorboardX;
#   - Megvii's voxel_pooling ops built with system gcc 9.4 + CUDA 11.6 for sm_86 (runs on the 4090s' sm_89).
# The coin3d env itself is not touched.
set -eo pipefail
OLD=/home/giyong/miniforge3/envs/coin3d
E=/home/giyong/miniforge3/envs/bevdepth_megvii
REPO=/data/giyong/baselines/BEVDepth_megvii
[ -d "$E" ] || cp -a /data/giyong/miniforge3/envs/coin3d /data/giyong/miniforge3/envs/bevdepth_megvii
cd "$E"
for f in $(grep -lI "$OLD" bin/* 2>/dev/null || true); do sed -i "s#$OLD#$E#g" "$f"; done
PY=$E/bin/python
$PY -m pip uninstall -y mmdet3d nuscenes-devkit diff-gaussian-rasterization || true
# pip leaves their /home/giyong/... lines in the copy's easy-install.pth (it matches the realpath /data/giyong/...)
: > lib/python3.8/site-packages/easy-install.pth
$PY -m pip install setuptools==59.5.0
$PY -m pip install nuscenes-devkit==1.1.10
$PY -m pip install /data/giyong/src/mmdet3d_rc4/mmdet3d-1.0.0rc4
$PY -m pip install "pip<24.1"  # pip >= 24.1 rejects pytorch-lightning 1.6.0's metadata ('torch (>=1.8.*)')
$PY -m pip install pytorch-lightning==1.6.0 torchmetrics==0.7.3 tensorboardX
cd "$REPO"
export CUDA_HOME=/home/giyong/opt/cuda-11.6 PATH=/home/giyong/opt/cuda-11.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.6 FORCE_CUDA=1
$PY setup.py develop
$PY -m pip list 2>/dev/null | grep -i -E "^(torch|torchvision|mmcv-full|mmdet|mmdet3d|mmsegmentation|pytorch-lightning|torchmetrics|nuscenes-devkit|numba|numpy|setuptools|BEVDepth) "
