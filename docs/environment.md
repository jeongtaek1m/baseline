# 환경

서버: RTX 4090 × 4 (compute capability 8.9), NVIDIA driver 535 (CUDA 12.2), gcc 10.5, CUDA toolkit `/usr/local/cuda-{11.1,11.3,…}`.

환경은 conda env 2개로 나눠요. 두 base가 요구하는 mmdet3d 버전(1.0.0rc4 vs 0.17.1)과 좌표계가 달라서 env를 공유할 수 없어요.

| env | 용도 | 만드는 스크립트 | 로그 |
|---|---|---|---|
| `gen2e-bevdepth` | BEVDepth, +PD-BEV, +UDGA (`bevdepth/`) | `tools/setup_env_bevdepth.sh` | `logs/setup_env_bevdepth*.log` |
| `gen2e-bevformer` | BEVFormer-S, +PD-BEV (`bevformer/`) | `tools/setup_env_bevformer.sh` | `logs/setup_env_bevformer.log` |

## gen2e-bevdepth

PCCR 공식 BEVDet Docker(`third_party/pccr/models/BEVDet/docker/Dockerfile`)와 같은 버전으로 맞췄어요. 그래야 BEVDet† 행과 비교할 수 있어요.

| 패키지 | 버전 |
|---|---|
| python | 3.8 (conda-forge) |
| torch / torchvision | 1.10.0+cu113 / 0.11.0+cu113 |
| mmcv-full | 1.5.3 (OpenMMLab 사전 빌드 wheel, cu113/torch1.10) |
| mmdet / mmsegmentation | 2.25.1 / 0.25.0 |
| mmdet3d | 1.0.0rc4 = `bevdepth/` editable 설치 (PD-BEV 트리) |
| numpy / numba | 1.20.3 / 0.53.0 |
| 기타 | nuscenes-devkit, lyft_dataset_sdk, networkx 2.2, trimesh 2.35.39, yapf 0.40.1, Pillow 9.5.0, timm 0.6.13, einops |

**확인한 것 (2026-11-22)**

- `pip check`에서 충돌이 없어요.
- mmdet3d가 `bevdepth/mmdet3d`에서 로드돼요.
- `bev_pool_v2`를 CUDA 11.3, `TORCH_CUDA_ARCH_LIST=8.6`으로 빌드했어요. 4090에서 forward/backward 결과가 CPU 기준값과 일치해요(최대 오차 1.2e-7).

## gen2e-bevformer

공식 BEVFormer 설치 가이드(`bevformer/docs/install.md`)를 따랐어요.

| 패키지 | 버전 |
|---|---|
| python | 3.8 (conda-forge) |
| torch / torchvision | 1.9.1+cu111 / 0.10.1+cu111 |
| mmcv-full | 1.4.0 (사전 빌드 wheel, cu111/torch1.9.0) |
| mmdet / mmsegmentation | 2.14.0 / 0.14.1 |
| mmdet3d | 0.17.1 = `third_party/mmdetection3d-v0.17.1` editable 설치 (git 제외, 스크립트가 다시 만듦) |
| detectron2 | 0.6 (가이드는 master인데 torch 1.9로 빌드되는 마지막 태그를 씀) |
| 기타 | timm 0.6.13, numpy 1.19.5, numba 0.48.0, yapf 0.40.1, nuscenes-devkit 1.1.9 |

**확인한 것 (2026-11-21):** `pip check`에서 충돌이 없어요. `mmcv.ops`(MSDA 포함)가 import되고, 4090에서 CUDA 연산이 돼요.

## RTX 4090 관련 메모

- **sm_89 미지원:** CUDA 11.1과 11.3은 sm_89를 타깃으로 빌드하지 못해요. 그래서 CUDA 확장은 sm_86으로 빌드해요(`TORCH_CUDA_ARCH_LIST=8.6`). 같은 major 아키텍처라 바이너리 호환으로 sm_89에서 그대로 돌아요. 실제 커널 실행으로 확인했어요.
- **PD-BEV의 사전 빌드 `.so`:** PD-BEV 원본에 커밋된 `bev_pool_v2_ext*.so`와 `build/`는 작성자 환경(GCC 5.4, CUDA 11.0)에서 빌드한 거예요. 지우고 다시 빌드했고, git에서도 제외했어요.
- **yapf:** 0.40.2 이상에서는 mmcv 1.x의 config 출력이 깨져요(`FormatCode(verify=...)`). 두 env 모두 0.40.1로 고정했어요.

## 설치 시 레포 원본 보호

업스트림 BEVDet `setup.py`의 `add_mim_extension()`은 `pip install -e .`을 할 때 `mmdet3d/.mim/{tools,configs}`를 지우고 심볼릭 링크로 바꿔요. PD-BEV는 그 자리에 미공개 스크립트를 커밋해 뒀기 때문에, 이 호출을 막았어요(커밋 `928c5b0`).
