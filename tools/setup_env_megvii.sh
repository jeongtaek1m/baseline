#!/usr/bin/env bash
# Conda env for the official Megvii BEVDepth code in ./megvii_bevdepth (vanilla BEVDepth row,
# user decision 2026-11-23). Separate from gen2e-bevdepth on purpose: that env has the PD-BEV
# tree installed as mmdet3d, whose CenterHead and pipelines PD-BEV edited in place, while the
# Megvii code imports mmdet3d's own CenterHead / SECONDFPN. This env gets pristine
# mmdet3d 1.0.0rc4 (the version Megvii's README asks for).
# Versions as gen2e-bevdepth (torch 1.10.0+cu113 / mmcv-full 1.5.3 / mmdet 2.25.1, known to run
# on the RTX 4090 with sm_86 builds); Megvii's README asks for torch 1.9.0, its requirements pin
# pytorch-lightning 1.6.0 and setuptools 59.5.0.
#
# Usage: bash tools/setup_env_megvii.sh [ENV_NAME]   (default: gen2e-megvii)
set -euo pipefail

ENV_NAME="${1:-gen2e-megvii}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="$(conda info --base)"

export CUDA_HOME=/usr/local/cuda-11.3
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="8.6"
export FORCE_CUDA=1
export MAX_JOBS="${MAX_JOBS:-16}"

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" -c conda-forge --override-channels python=3.8
fi
conda activate "${ENV_NAME}"

pip install --upgrade "pip<24.1"
pip install setuptools==59.5.0 wheel cython numpy==1.20.3
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
pip install mmdet==2.25.1 mmsegmentation==0.25.0
# pristine mmdet3d 1.0.0rc4 (compiles its CUDA ops)
pip install mmdet3d==1.0.0rc4
# Megvii BEVDepth requirements.txt (torch/torchvision pins replaced by the versions above)
pip install numba==0.53.0 numpy==1.20.3 nuscenes-devkit opencv-python-headless pandas \
  pytorch-lightning==1.6.0 scikit-image scipy tensorboardX "torchmetrics<0.11" Pillow==9.5.0 \
  yapf==0.40.1

( cd "${REPO_ROOT}/megvii_bevdepth" && python setup.py develop )

python - <<'EOF'
import torch, mmcv, mmdet, mmdet3d, pytorch_lightning as pl
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
print('mmcv', mmcv.__version__, '| mmdet', mmdet.__version__, '| mmdet3d', mmdet3d.__version__, 'from', mmdet3d.__file__)
print('pytorch_lightning', pl.__version__)
from mmcv.ops import nms_rotated  # noqa: F401
from bevdepth.ops.voxel_pooling_train import voxel_pooling_train  # noqa: F401
print('mmcv.ops and bevdepth voxel pooling ops OK')
EOF
echo "env ${ENV_NAME} ready"
