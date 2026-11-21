#!/usr/bin/env bash
# Conda env for BEVFormer-S (+ PD-BEV port).
# Follows the official guide (bevformer/docs/install.md, fundamentalvision/BEVFormer @ 66b65f3)
# with pins that keep mmcv 1.x working today:
#   torch 1.9.1+cu111 / mmcv-full 1.4.0 / mmdet 2.14.0 / mmseg 0.14.1 / mmdet3d 0.17.1
# RTX 4090 (sm_89): CUDA 11.1 cannot target sm_89, so ops are built for sm_86; they run on sm_89
# through binary compatibility (same major arch).
#
# Usage: bash tools/setup_env_bevformer.sh [ENV_NAME]   (default: gen2e-bevformer)
set -euo pipefail

ENV_NAME="${1:-gen2e-bevformer}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="$(conda info --base)"
MMDET3D_DIR="${REPO_ROOT}/third_party/mmdetection3d-v0.17.1"   # gitignored, re-created by this script

export CUDA_HOME=/usr/local/cuda-11.1
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
pip install setuptools==59.5.0 wheel cython
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 \
  -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.14.0 mmsegmentation==0.14.1

if [ ! -d "${MMDET3D_DIR}" ]; then
  git clone --branch v0.17.1 --depth 1 https://github.com/open-mmlab/mmdetection3d.git "${MMDET3D_DIR}"
fi
( cd "${MMDET3D_DIR}" && pip install -v -e . )

pip install einops fvcore seaborn iopath==0.1.9 timm==0.6.13 typing-extensions==4.5.0 pylint \
  ipython==8.12 numpy==1.19.5 matplotlib==3.5.2 numba==0.48.0 pandas==1.4.4 \
  scikit-image==0.19.3 setuptools==59.5.0 yapf==0.40.1 nuscenes-devkit==1.1.9
# detectron2: official guide installs master; v0.6 is the last tag that builds with torch 1.9.
pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'

python - <<'EOF'
import torch, mmcv, mmdet, mmseg, mmdet3d
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
print('arch list', torch.cuda.get_arch_list())
print('mmcv', mmcv.__version__, '| mmdet', mmdet.__version__, '| mmseg', mmseg.__version__, '| mmdet3d', mmdet3d.__version__)
from mmcv.ops import MultiScaleDeformableAttention  # noqa: F401  (compiled op)
print('mmcv.ops OK')
if torch.cuda.is_available():
    x = torch.randn(64, 64, device='cuda'); print('cuda matmul OK', float((x @ x).sum()) == float((x @ x).sum()))
EOF
echo "env ${ENV_NAME} ready"
