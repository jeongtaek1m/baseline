#!/usr/bin/env bash
# Conda env for the BEVDepth family (vanilla BEVDepth, +PD-BEV, +UDGA), built on the PD-BEV repo
# (BEVDet fork, mmdet3d 1.0.0rc4) in ./bevdepth.
# Versions follow the PCCR benchmark's BEVDet Docker image (third_party/pccr/models/BEVDet/docker/Dockerfile)
# so that our BEVDepth numbers are comparable with the BEVDet† row:
#   torch 1.10.0+cu113 / mmcv-full 1.5.3 / mmdet 2.25.1 / mmseg 0.25.0 / numba 0.53.0 / numpy 1.20.3
# RTX 4090 (sm_89): CUDA 11.3 cannot target sm_89, so ops are built for sm_86 (binary compatible).
#
# Usage: bash tools/setup_env_bevdepth.sh [ENV_NAME]   (default: gen2e-bevdepth)
# The editable install of ./bevdepth (compiles bev_pool_v2) runs only if bevdepth/setup.py exists.
set -euo pipefail

ENV_NAME="${1:-gen2e-bevdepth}"
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
# runtime deps of mmdet3d 1.0.0rc4 (bevdepth/requirements/runtime.txt) + PCCR Docker pins
pip install lyft_dataset_sdk "networkx>=2.2,<2.3" numba==0.53.0 numpy==1.20.3 nuscenes-devkit plyfile \
  scikit-image tensorboard "trimesh>=2.35.39,<2.35.40" yapf==0.40.1 Pillow==9.5.0 pycuda
# PD-BEV imports these at import time (mmdet3d/models/backbones/__init__.py -> context_cluster*.py,
# which needs the timm<=0.6 API); they are missing from its requirements/.
pip install timm==0.6.13 einops

if [ -f "${REPO_ROOT}/bevdepth/setup.py" ]; then
  ( cd "${REPO_ROOT}/bevdepth" && pip install -v -e . )
else
  echo "NOTE: ${REPO_ROOT}/bevdepth/setup.py not found -> skipped the editable install (bev_pool_v2 not built yet)"
fi

python - <<'EOF'
import torch, mmcv, mmdet, mmseg
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
print('arch list', torch.cuda.get_arch_list())
print('mmcv', mmcv.__version__, '| mmdet', mmdet.__version__, '| mmseg', mmseg.__version__)
from mmcv.ops import nms_rotated  # noqa: F401  (compiled op)
print('mmcv.ops OK')
try:
    import mmdet3d
    print('mmdet3d', mmdet3d.__version__, 'from', mmdet3d.__file__)
    from mmdet3d.ops.bev_pool_v2.bev_pool import bev_pool_v2  # noqa: F401
    print('bev_pool_v2 OK')
except Exception as e:  # editable install may not have happened yet
    print('mmdet3d / bev_pool_v2 not available yet:', repr(e))
EOF
echo "env ${ENV_NAME} ready"
