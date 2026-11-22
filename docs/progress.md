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
- **데이터:** PCCR 14개 리그를 모두 받았어요. sha256 검증을 통과했고, 압축을 풀면 193 GB예요. 오류는 없었어요. `/data2/PCCR`에 있고 `data/pccr` 링크로 접근해요.
- **결정:** 학습도 7개 클래스로 하고(traffic 2종 제외), BEVDepth 계열 학습 설정은 PCCR BEVDet 기준으로 정했어요([decisions.md](decisions.md)).
- **박스 z 확인 (사용자 요청):** `tools/pccr_box_z_check.py`로 14개 리그를 확인했어요. 자세한 결과는 [pccr_dataset.md](pccr_dataset.md) "박스 z 확인"에 있어요.
  - 박스 z는 중심이고 정확해요. 차량은 지면에 닿아 있고, 보행자는 지면에서 3 cm 위예요(LiDAR 대조).
  - ego 원점은 4개 차종 모두 지면이에요. 그래서 ego 좌표계를 쓰는 BEVDepth 계열은 고칠 게 없어요.
  - LiDAR 좌표계는 리그마다 지면 높이가 1.50~4.80 m로 달라요. BEVFormer-S에서 어느 좌표계를 쓸지 새 결정 항목으로 올렸어요.
  - PCCR의 PETR·BEVFusion 로더는 z를 바닥으로 읽어서 학습 박스가 h/2 떠 있어요. † 지표에는 영향이 없어요.
- **vanilla BEVDepth 포팅 (포팅 계획 0-A, 0-B, 1장).** 자세한 내용은 [porting_plan.md](porting_plan.md)의 "구현" 단락에 있어요. 커밋은 `ea224a7`~`55b52cc`예요.
  - PCCR 어댑터를 그대로 옮기고, 원본 BEVDet 클래스 4개를 바이트 단위로 복원했어요. `tools/check_bevdepth_upstream.py`로 확인해요.
  - `cams=None` 처리(`PrepareImageInputsPCCR`)와 속도 없는 head(`CenterHeadNoVel`)를 추가했어요.
  - PD-BEV 도구의 디버그 코드를 정리했어요(anomaly detection, `/mnt/cfs` 덤프).
  - 설정은 `bevdepth/configs/gen2e/bevdepth-r50-pccr.py`예요. PCCR BEVDet 설정(dev3.0)을 dev2.0 파이프라인으로 옮기고, 업스트림 BEVDepth 요소를 더하고, 7개 클래스로 바꿨어요.
  - 스모크 테스트(R1 mini, 1 epoch)에서 학습과 평가가 끝까지 돌았어요. depth 라벨 정렬도 이미지로 확인했어요.
- **vanilla BEVDepth R1 학습을 시작했어요.** GPU 0·1을 쓰고, 출력은 `bevdepth/work_dirs/bevdepth-r50-pccr/R1/`, 로그는 `logs/train_bevdepth-r50-pccr_R1.log`예요. epoch당 473 iter, 24 epoch이에요.

- **vanilla BEVDepth 결과:** 학습은 14:50에 끝났어요(R1 val 33.5). 14개 리그 test 평가 결과 7-class mAP가 R1 30.4예요. 행 전체는 [results.md](results.md)에 있어요.
  - BEVDet† 대비 R1과 가까운 리그는 5~7점 높고, 먼 리그(R3, R6, R7)는 2~3점 낮아요.
- **PCCR 공식 모델의 좌표계 조사 (사용자 요청):** [pccr_dataset.md](pccr_dataset.md) "PCCR 공식 모델의 좌표계"에 정리했어요.
  - BEVDet만 ego 좌표계이고 PETR·BEVFusion·Fast-BEV는 LiDAR 좌표계예요. 리그별 높이 보정은 없어요.
  - LiDAR 좌표계에서 높이 여러 층을 샘플링하는 PETR·Fast-BEV만 R6·R9에서 0.0이에요.
  - 부수적으로, PETR·BEVFusion·Fast-BEV 어댑터는 평가 박스의 w/l이 뒤바뀌고 yaw가 반전돼요(ASE 0.73, AOE ≈ π/2). mAP에는 영향이 없어요.
- **결정:** BEVFormer-S는 LiDAR 좌표계(A)로 학습해요([decisions.md](decisions.md)).

## 다음 할 일

1. `bevformer/`에 PCCR을 통합하고(0-C), BEVFormer-S를 R1에서 학습해요(4장). 학습 설정(배치·lr, 해상도, 사전학습 가중치)은 사용자 확인이 필요해요.
2. 남은 결정에 답을 받아요([decisions.md](decisions.md)의 "확인 필요"). 기존 BEVFormer-S 코드, PD-BEV head 구성, 공식 코드 버그 처리가 남았어요.
3. PD-BEV(2장, 5장)와 UDGA(3장)를 이식해요.
