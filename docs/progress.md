# 진행 기록

## 2026-11-21

- **Baseline 레포 확보:** PD-BEV(`EnVision-Research/Generalizable-BEV`)와 UDGA(`SAITPublic/UDGA`)를 클론했어요. 사용자가 말한 "ugbev"는 UDGA로 확인했어요.
- **데이터:** 표의 데이터셋이 PCCR(ECCV 2026)이라는 걸 확인했어요. 서버에는 없었어요.
  - `tools/download_pccr.sh`(HF 리비전 고정 + sha256 검증)로 `/data2/PCCR`에 받기 시작했어요.
- **표 지표 확인:** 표의 † 행 4개가 PCCR 공식 결과의 **7-class mAP**(traffic 2종 제외, test split)와 0.05 이내로 일치해요(`tools/pccr_metric.py verify-dagger`).
- **서버 정리 (사용자 요청)**
  - `/data2/TCAR_DATA`(353G, sudo)와 `/data2/TCAR_DATA_bag`(134G)을 지웠어요.
  - 홈의 다른 프로젝트 8개와 conda env 9개를 지웠고, pip·HF 캐시도 비웠어요.
  - foundation과 tcar_bevdet의 소스 코드만 `~/backup/*.tar.gz`로 남겼어요.
- **환경:** `gen2e-bevformer`, `gen2e-bevdepth`(코어)를 만들었어요.
- **코드 분석 워크플로:** 분석 에이전트 5개(PD-BEV, UDGA, BEVFormer, BEVDepth base 후보, UniDrive/CoIn3D)와 교차 검증 에이전트 4개를 돌렸어요. 결과는 [porting_plan.md](porting_plan.md)에 정리했어요.

## 2026-11-22

- **BEVDepth base:** PD-BEV 트리가 원본 BEVDet 클래스(`PointToMultiViewDepth` 등)를 직접 고친 사실을 확인해서 사용자에게 다시 물었어요. 사용자가 **PD-BEV 트리를 그대로 쓰기로** 재확인했어요. vanilla BEVDepth를 위해 원본 클래스를 별도로 복원하는 작업을 계획에 넣었어요.
- **레포 정리 + git 초기화**
  - `PD-BEV/` → `bevdepth/`, `UDGA/` → `third_party/UDGA/`로 옮겼어요. 공식 BEVFormer(`bevformer/`)와 PCCR 벤치마크(`third_party/pccr/`)를 추가했어요.
  - `d87eefc`: 업스트림 원본 스냅샷이에요(수정 없음).
  - `5efe61f`: `bevdepth/`에 `setup.py`, `setup.cfg`, `requirements.txt`, `MANIFEST.in`, `LICENSE`를 BEVDet @2ae6a02에서 그대로 복원했어요(blob 해시 일치).
  - `928c5b0`: editable 설치를 할 때 PD-BEV의 `.mim`이 지워지지 않게 했어요.
- **환경:** `gen2e-bevdepth`에 PD-BEV 트리를 editable로 설치하고 `bev_pool_v2`를 빌드했어요. 4090에서 forward/backward를 확인했어요.
- **데이터:** PCCR 리그 14개 중 12개를 받았어요(R8, R9 진행 중). 오류는 없어요.

## 다음 할 일

1. 사용자 확인이 필요한 결정에 답을 받아요([decisions.md](decisions.md)의 "확인 필요").
2. `bevdepth/`에 PCCR 데이터셋 어댑터와 `cams=None` 처리를 넣어요(포팅 계획 0-A).
3. vanilla BEVDepth용 원본 클래스를 복원하고 설정을 만들어요(0-B, 1장). 그다음 R1 학습(GPU 2장)과 14개 리그 평가를 해요.
4. `bevformer/`에 PCCR을 통합하고(0-C), BEVFormer-S 재현을 확인해요(4장).
5. PD-BEV(2장, 5장)와 UDGA(3장)를 이식해요.
