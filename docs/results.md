# 결과

모든 수치는 **7-class mAP(%)**예요. R1 train으로 학습하고, 각 리그의 **test** split으로 평가했어요. 지표 정의는 [evaluation_protocol.md](evaluation_protocol.md)에 있어요.

| 행 | R1 | R1-c6 | R1-c10 | R1-f | R1-r | R1-t | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **BEVDepth (vanilla, 우리)** | 30.4 | 27.8 | 33.9 | 0.2 | 30.8 | 29.1 | 0.1 | 5.7 | 31.1 | 0.0 | 6.2 | 9.2 | 5.5 | 4.1 |
| **BEVDepth + PD-BEV-A (우리)** | 27.8 | 25.1 | 29.6 | 15.3 | 27.5 | 27.4 | 18.7 | 20.8 | 28.5 | 4.3 | 11.9 | 16.4 | 11.4 | 7.3 |
| BEVDepth + PD-BEV-B (우리, 섭동 범위를 PCCR 리그에 맞춤) | 27.8 | 25.7 | 30.1 | 16.3 | 30.1 | 30.4 | 19.7 | 22.3 | 29.1 | 4.8 | 14.1 | 19.8 | 12.3 | 8.8 |
| **BEVDepth + UDGA (우리, 처음 설정)** | 33.0 | 30.0 | 35.9 | 21.3 | 33.2 | 34.4 | 23.6 | 25.9 | 34.1 | 7.0 | 13.1 | 22.4 | 15.9 | 6.9 |
| BEVDepth + UDGA-Lp (우리, 미분 가능한 L_p) | 34.1 | 30.7 | 36.8 | 24.1 | 35.1 | 35.5 | 24.7 | 29.2 | 35.2 | 8.2 | 16.4 | 23.4 | 17.7 | 10.0 |
| BEVDepth + UDGA-Ld (우리, LiDAR depth) | 32.1 | 29.2 | 35.2 | 20.2 | 34.1 | 31.4 | 20.2 | 22.4 | 32.1 | 6.1 | 9.4 | 18.4 | 14.6 | 4.2 |
| **BEVFormer-S (우리)** | 34.5 | 32.8 | 34.2 | 24.7 | 33.1 | 33.0 | 23.4 | 30.0 | 35.2 | 8.5 | 18.3 | 22.5 | 17.7 | 13.0 |
| **BEVDepth (Megvii 공식 코드, 공저자 base, 우리)** | 32.1 | 29.3 | 35.4 | 0.6 | 33.3 | 25.9 | 0.6 | 10.4 | 31.0 | 0.0 | 7.1 | 12.4 | 6.5 | 3.2 |
| **BEVDepth + PD-BEV-A (Megvii base, 우리)** | 33.6 | 31.3 | 35.8 | 19.9 | 34.2 | 33.0 | 23.3 | 25.4 | 34.7 | 6.0 | 14.4 | 22.0 | 15.2 | 8.9 |
| BEVDepth + PD-BEV-B (Megvii base, 우리, 섭동 범위를 PCCR 리그에 맞춤) | 34.0 | 31.3 | 36.3 | 19.3 | 34.6 | 34.8 | 23.1 | 25.4 | 34.6 | 5.6 | 13.4 | 21.7 | 15.2 | 8.7 |
| BEVDepth + UDGA-Lp (Megvii base, 우리, 논문 최종 행) | 35.0 | 31.7 | 36.1 | 20.8 | 35.2 | 34.2 | 21.3 | 22.0 | 35.6 | 5.0 | 9.7 | 16.8 | 15.7 | 4.7 |
| BEVDepth + UDGA-Ld (Megvii base, 우리, 공개 config 구성) | 32.7 | 30.2 | 35.9 | 18.4 | 34.6 | 30.8 | 17.3 | 19.0 | 32.9 | 4.9 | 7.1 | 14.2 | 14.8 | 2.8 |
| (참고) BEVDet† | 25.2 | 23.2 | 27.1 | 0.5 | 24.7 | 21.7 | 1.3 | 9.0 | 25.5 | 0.0 | 8.2 | 11.2 | 4.9 | 5.1 |
| (참고) BEVFusion-C† | 26.0 | 24.0 | 27.1 | 0.9 | 24.0 | 21.7 | 2.2 | 14.6 | 25.7 | 0.0 | 10.6 | 12.8 | 3.6 | 6.5 |
| (참고) Fast-BEV† | 12.9 | 12.5 | 19.2 | 6.0 | 17.6 | 18.3 | 7.0 | 9.0 | 6.1 | 1.2 | 0.0 | 9.0 | 1.8 | 0.0 |
| (참고) PETR† | 21.1 | 18.8 | 21.0 | 2.3 | 18.2 | 13.4 | 1.8 | 10.7 | 17.0 | 0.1 | 0.0 | 5.8 | 1.8 | 0.0 |

† 행은 PCCR 공식 결과에서 다시 계산한 값이에요(`python3 tools/pccr_metric.py verify-dagger`). 9개 클래스로 학습한 모델이에요.

## 사용자 표 형식 (2026-11-23)

사용자 표와 같은 열이에요: 14개 리그, OOD 평균(R1 제외 13개 평균), 최저(13개 중), 하락률(1 − OOD 평균 / R1), 14개 평균, H(R1과 OOD 평균의 조화평균). 사용자 표의 BEVDepth [42] 값으로 이 다섯 열의 계산식을 재현했어요(16.595 / 0.000 / 55.0% / 18.042 / 22.885).

| 행 | R1 | R1-c6 | R1-c10 | R1-f | R1-r | R1-t | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | OOD 평균 | 최저 | 하락률 | 14개 평균 | H | 좌표계 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BEVDepth (우리) | 30.430 | 27.842 | 33.852 | 0.158 | 30.844 | 29.099 | 0.142 | 5.705 | 31.112 | 0.000 | 6.246 | 9.193 | 5.534 | 4.109 | 14.141 | 0.000 | 53.5% | 15.305 | 19.309 | ego (차량 지면 원점) |
| + PD-BEV-A (우리) | 27.832 | 25.118 | 29.605 | 15.290 | 27.523 | 27.370 | 18.653 | 20.759 | 28.522 | 4.284 | 11.924 | 16.379 | 11.428 | 7.340 | 18.784 | 4.284 | 32.5% | 19.430 | 22.430 | ego (차량 지면 원점) |
| + PD-BEV-B (우리, 리그 맞춤 섭동) | 27.812 | 25.705 | 30.140 | 16.262 | 30.105 | 30.421 | 19.665 | 22.345 | 29.081 | 4.787 | 14.081 | 19.752 | 12.340 | 8.760 | 20.265 | 4.787 | 27.1% | 20.804 | 23.446 | ego (차량 지면 원점) |
| + UDGA (우리, 처음 설정) | 32.961 | 29.975 | 35.857 | 21.284 | 33.197 | 34.423 | 23.621 | 25.911 | 34.104 | 6.979 | 13.124 | 22.414 | 15.852 | 6.927 | 23.359 | 6.927 | 29.1% | 24.045 | 27.342 | ego (차량 지면 원점) |
| + UDGA-Lp (우리) | 34.144 | 30.720 | 36.846 | 24.125 | 35.095 | 35.499 | 24.660 | 29.223 | 35.197 | 8.183 | 16.431 | 23.420 | 17.694 | 9.987 | 25.160 | 8.183 | 26.3% | 25.802 | 28.972 | ego (차량 지면 원점) |
| + UDGA-Ld (우리) | 32.055 | 29.240 | 35.190 | 20.193 | 34.057 | 31.417 | 20.213 | 22.406 | 32.129 | 6.073 | 9.350 | 18.357 | 14.574 | 4.155 | 21.335 | 4.155 | 33.4% | 22.101 | 25.619 | ego (차량 지면 원점) |
| BEVFormer-S (우리) | 34.536 | 32.788 | 34.227 | 24.738 | 33.103 | 32.966 | 23.359 | 29.967 | 35.158 | 8.527 | 18.305 | 22.524 | 17.660 | 13.002 | 25.102 | 8.527 | 27.3% | 25.776 | 29.073 | 차량 고정 (ego 축, 원점 = R1 LiDAR 장착점) |
| BEVDepth (Megvii 공식, 공저자 base, 우리) | 32.137 | 29.339 | 35.448 | 0.609 | 33.292 | 25.884 | 0.583 | 10.351 | 30.979 | 0.000 | 7.149 | 12.431 | 6.463 | 3.182 | 15.055 | 0.000 | 53.2% | 16.275 | 20.504 | ego (차량 지면 원점) |
| + PD-BEV-A (Megvii base, 우리) | 33.630 | 31.305 | 35.778 | 19.919 | 34.195 | 32.950 | 23.302 | 25.376 | 34.651 | 6.036 | 14.386 | 21.996 | 15.230 | 8.884 | 23.385 | 6.036 | 30.5% | 24.117 | 27.587 | ego (차량 지면 원점) |
| + PD-BEV-B (Megvii base, 우리, 리그 맞춤 섭동) | 33.996 | 31.255 | 36.288 | 19.268 | 34.572 | 34.759 | 23.057 | 25.416 | 34.642 | 5.605 | 13.447 | 21.701 | 15.224 | 8.704 | 23.380 | 5.605 | 31.2% | 24.138 | 27.706 | ego (차량 지면 원점) |
| + UDGA-Lp (Megvii base, 우리) | 34.989 | 31.729 | 36.126 | 20.824 | 35.228 | 34.243 | 21.322 | 22.027 | 35.647 | 5.024 | 9.675 | 16.803 | 15.665 | 4.695 | 22.231 | 4.695 | 36.5% | 23.143 | 27.188 | ego (차량 지면 원점) |
| + UDGA-Ld (Megvii base, 우리) | 32.725 | 30.235 | 35.865 | 18.402 | 34.583 | 30.776 | 17.313 | 18.974 | 32.946 | 4.933 | 7.075 | 14.226 | 14.755 | 2.815 | 20.223 | 2.815 | 38.2% | 21.116 | 24.998 | ego (차량 지면 원점) |
| (참고) 사용자 표 BEVDepth [42] | 36.850 | 33.970 | 40.260 | 0.130 | 37.720 | 34.570 | 0.090 | 3.830 | 37.180 | 0.000 | 5.740 | 8.780 | 8.750 | 4.720 | 16.595 | 0.000 | 55.0% | 18.042 | 22.885 | ego (차량 지면 원점) |

- 우리 BEVDepth는 [42]보다 R1 계열 리그(R1, R1-c6, R1-c10, R1-r, R1-t, R4)에서 5.5~6.9점 낮아요. 먼 리그는 비슷하고, 하락률도 비슷해요.
- 평가 체인과 학습 기하는 `tools/check_bevdepth_pccr.py`로 확인했어요([progress.md](progress.md) 2026-11-23). [42]를 어떻게 학습했는지는 몰라요.
- 표의 BEVDepth 행을 바꿀지는 사용자가 다시 검증한 뒤 정해요.
- 값은 `metrics_summary.json`의 7개 클래스 AP를 평균해서 계산했어요.

## BEVDepth (vanilla)

- **날짜:** 2026-11-22
- **설정:** `bevdepth/configs/gen2e/bevdepth-r50-pccr.py`
  - 7개 클래스, 24 epoch, GPU 2장 × 4샘플이에요. 나머지는 [porting_plan.md](porting_plan.md) 1장을 보세요.
- **체크포인트:** `latest.pth`(= `epoch_24.pth`, EMA 아님)
  - PCCR `sequential_test.sh`와 같은 선택이에요. 중간 epoch 중에서 고르지 않았으니 test 정보가 새지 않아요.
- **학습 중 R1 val (epoch 24):** 33.5 mAP
- **실행 명령**

  ```bash
  bash tools/bevdepth_train.sh configs/gen2e/bevdepth-r50-pccr.py 0,1
  ```

  ```bash
  bash tools/bevdepth_test_rigs.sh bevdepth/work_dirs/bevdepth-r50-pccr/R1 0,1
  ```

- **원시 결과:** `bevdepth/work_dirs/bevdepth-r50-pccr/R1/test_on_<RIG>_test/` (git 제외)

**BEVDet†와 비교**

- R1과 가까운 리그(R1-c6, R1-c10, R1-r, R1-t, R4)에서는 5~7점 높아요.
- 먼 리그(R3, R6, R7)에서는 2~3점 낮아요.
- R1-f, R2, R5는 모든 모델이 0 근처예요.
- 박스 크기·방향 오차는 정상이에요. R1 test에서 car ASE 0.22, AOE 0.69예요.

## BEVDepth + PD-BEV-A

- **날짜:** 2026-11-23
- **설정:** `bevdepth/configs/gen2e/pdbev-r50-pccr-A.py`
  - vanilla 구조에 PD-BEV 구성요소를 더했어요. 섭동 범위는 공식 값이에요(pitch·roll ±0.04 rad, yaw ±0.4 rad, xyz ±2 m).
  - 공식 코드 버그 16개를 고쳤어요([bug_ledger.md](bug_ledger.md)).
- **체크포인트:** `latest.pth`(= `epoch_24.pth`, EMA 아님). vanilla와 같은 규칙이에요.
- **학습 중 R1 val (epoch 24):** 29.5 mAP
- **실행 명령**

  ```bash
  bash tools/bevdepth_train.sh configs/gen2e/pdbev-r50-pccr-A.py 0,1
  ```

  ```bash
  bash tools/bevdepth_test_rigs.sh bevdepth/work_dirs/pdbev-r50-pccr-A/R1 0,1
  ```

- **원시 결과:** `bevdepth/work_dirs/pdbev-r50-pccr-A/R1/test_on_<RIG>_test/` (git 제외)

**vanilla와 비교**

- R1 계열(R1, R1-c6, R1-c10, R1-r, R1-t, R4)은 1.7~4.3점 낮아요.
- 화각이나 차종이 다른 리그는 모두 올랐어요: R1-f 0.2 → 15.3, R2 0.1 → 18.7, R3 5.7 → 20.8, R5 0.0 → 4.3, R6 6.2 → 11.9, R7 9.2 → 16.4, R8 5.5 → 11.4, R9 4.1 → 7.3.
- OOD 평균은 14.1 → 18.8, 하락률은 53.5% → 32.5%예요.

## BEVFormer-S

- **날짜:** 2026-11-23
- **설정:** `bevformer/projects/configs/gen2e/bevformer_small_pccr.py`
  - bevformer_small 구조, 단일 프레임, CAN bus 끔, R101(ImageNet) 초기화, 입력 1280×720, 배치 8, 24 epoch이에요.
  - 차량 고정 좌표계를 써요. 축은 ego와 같고, 원점은 모든 차량에서 R1 LiDAR 장착점이에요.
- **체크포인트:** `latest.pth`(= `epoch_24.pth`). 다른 행과 같은 규칙이에요.
- **학습 중 R1 val (epoch 24):** 41.7 mAP. epoch 19~24에서 40.5~41.7로 수렴했어요.
- **실행 명령**

  ```bash
  bash tools/bevformer_train.sh projects/configs/gen2e/bevformer_small_pccr.py 2,3
  ```

  ```bash
  bash tools/bevformer_test_rigs.sh bevformer/work_dirs/bevformer_small_pccr/R1 2,3
  ```

- **원시 결과:** `bevformer/work_dirs/bevformer_small_pccr/R1/test_on_<RIG>_test/` (git 제외)

**독립 검증 (2026-11-23)**

기존 표의 BEVFormer-S 행보다 훨씬 높아서, 표에 넣기 전에 검증 워크플로를 돌렸어요. 에이전트 5개가 CPU로 37분 걸렸고, 원자료는 [analysis/2026-11-23/verify_bevformer_row.json](analysis/2026-11-23/verify_bevformer_row.json)이에요. 판정은 **계산된 값은 신뢰할 수 있음(비교 시 주의점 있음)**이에요. 심각도 높음·중간인 문제는 없었어요.

- **값 재현:** 저장된 `metrics_summary.json`에서 14개 리그 모두 다시 계산돼요(최대 차이 0.0005). 원시 `results_nusc.json`을 PCCR 평가 경로로 다시 평가해도 같아요(리그당 94개 값, 차이 0). devkit을 쓰지 않는 독립 평가기 2개도 클래스·거리 기준별 AP가 모두 같아요.
- **리그별 데이터:** 결과마다 그 리그의 test 토큰 1800개와 정확히 같아요. 리그끼리 공유하는 토큰은 없어요. 리그마다 15개 샘플을 CPU로 다시 추론하면 저장된 박스와 8.2 cm, 점수 0.0032 이내로 같아요. R1 이미지를 대신 넣으면 11~19 m 어긋나요.
- **체크포인트·split:** 14개 리그 모두 epoch_24예요. 학습은 R1 train(3780)만 썼고, epoch별 평가는 R1 val(1020)로 했어요. train/val/test는 장면·토큰·이미지가 겹치지 않아요. test 도시(Town03/06/07)는 학습에 없어요.
- **누수 없음:** test 입력은 이미지와 카메라 행렬뿐이에요(CAN bus 0). GT에서 1 cm 안에 있는 예측이 거의 없어요(R1 0.08%). R1 val(41.7)이 test(34.5)보다 높아요.
- **좌표계:** GT를 예측으로 넣으면 R1, R4, R5, R6, R9, R1-f에서 mAP 100, 위치·크기 오차 0이에요. R6·R9 예측 차량의 바닥 높이도 GT와 같아요(−1.85 m 근처).

**비교할 때 주의할 점**
- **점수 하한:** BEVFormer는 공식 설정대로 샘플당 박스 300개를 점수 하한 없이 내요. BEVDepth 계열은 공식 CenterPoint 설정대로 0.1 미만을 버려요. BEVFormer에 0.1 하한을 걸면 R1 33.2(−1.3), R5 8.1, R9 12.3이에요.
- **좌표계:** 차량 고정 좌표계는 etron 리그 9개에서는 LiDAR 좌표계와 같아요. R4/R5/R8(audi), R6(소방차), R9(트럭)에서는 PETR†·Fast-BEV†가 쓴 LiDAR 좌표계와 달라요. 표에 좌표계를 적어야 해요. 그 차이의 크기는 재지 않았어요(GPU 추론이 필요해요).
- **학습 1회:** seed에 따른 편차는 몰라요.
- **방향 오차:** mAOE가 1.02~1.27로 높아요. GT를 예측으로 넣은 왕복 검사에서 AOE가 0.006 이하라서, 변환 문제가 아니라 모델 자체의 오차예요. mAP에는 영향이 없어요.

**기존 표 행과의 차이:** 기존 BEVFormer-S 행(R1 21.1, R1-c6 19.6, R1-c10 26.2)보다 13.4, 13.2, 8.0점 높아요. 이 세 리그에서는 좌표계가 원인이 아니에요(etron 리그라 같은 좌표계예요). 기존 행의 코드와 설정을 모르니 원인을 특정할 수 없어요. 우리 설정에서 확인된 사실은 이래요.
- 입력이 원본 1280×720이고, R101에 DCN이 있어요.
- vanilla BEVDepth 대비 R1 이득은 작은 클래스에 몰려 있어요: bicycle 49.9 vs 38.7, adult 39.1 vs 27.5, child 9.8 vs 2.4.
- car와 bus는 오히려 낮아요: 40.2 / 24.2 vs 43.0 / 27.9.

## BEVDepth + PD-BEV-B

- **날짜:** 2026-11-23
- **설정:** `bevdepth/configs/gen2e/pdbev-r50-pccr-B.py`. PD-BEV-A와 섭동 범위만 달라요: yaw ±22.5°, pitch ±20°, roll ±0.04 rad, x −2.9~+5.3 m, y −0.6~+0.8 m, z −0.5~+2.7 m.
- **주의:** 섭동 범위를 PCCR test 리그의 카메라 배치에서 계산했어요. 테스트 리그 정보를 쓴 설정이라 표에서 따로 표시해야 해요.
- **체크포인트:** `latest.pth`(= `epoch_24.pth`). **학습 중 R1 val (epoch 24):** 29.7 mAP
- **결과:** OOD 평균 20.3으로 PD-BEV-A(18.8)보다 1.5점 높아요. R1은 같아요(27.8). 카메라 위치·방향이 바뀐 리그(R1-r +2.6, R1-t +3.1, R6 +2.2, R7 +3.4)에서 주로 올랐어요.

## BEVDepth + UDGA (처음 설정)

- **날짜:** 2026-11-23
- **설정:** `bevdepth/configs/gen2e/udga-r50-pccr.py`. LiDAR depth 없음, L_ov(대응 칸만, 가중치 1), L_p는 공개 코드대로(그래디언트 0), pc400 virtual depth bin [1, 90), ext aug예요([decisions.md](decisions.md)).
- **체크포인트:** `latest.pth`(= `epoch_24.pth`). **학습 중 R1 val (epoch 24):** 35.3 mAP(vanilla 33.5)
- **결과:** R1 33.0으로 vanilla(30.4)보다 높아요. 화각이 바뀐 리그에서도 무너지지 않아요(R1-f 21.3, R2 23.6, R3 25.9). OOD 평균은 23.4로 PD-BEV-A·B보다 높아요.
- **depth 분포가 무너졌어요:** epoch 4에서 depth 분포가 거의 균일했어요(최대 확률 0.022, 균일 0.011, vanilla 0.47). L_ov는 모든 logit이 같으면 0이 되기 때문이에요. 그러니 이 행은 사실상 depth 예측 없이 virtual depth 범위 전체에 feature를 고르게 펼치는 모델이에요. 논문 표 4에서 LiDAR depth 없는 L_ov 행이 L_d 행보다 일반화가 좋았던 것과 같은 방향이에요.
- **추가 변형 두 개를 학습 중이에요**(2026-11-23 결정): UDGA-Lp(논문 식 4의 미분 가능한 L_p)와 UDGA-Ld(LiDAR depth, virtual depth 타깃). 세 설정을 비교해서 표에 넣을 행을 골라요.

## BEVDepth + UDGA-Lp / UDGA-Ld (변형, 2026-11-23)

- **설정:** `bevdepth/configs/gen2e/udga-r50-pccr-Lp.py`(L_p를 논문 식 4대로 미분 가능하게, LiDAR depth 없음), `udga-r50-pccr-Ld.py`(LiDAR depth를 virtual depth 타깃으로, L_p는 공개 코드). 나머지는 처음 설정과 같아요. 체크포인트는 둘 다 `epoch_24.pth`예요.
- **R1 val (epoch 24):** UDGA-Lp 36.3, UDGA-Ld 37.1
- **결과**
  - UDGA-Lp는 R1 34.1, OOD 평균 25.2로 세 UDGA 설정 중 가장 높아요.
  - UDGA-Ld는 R1 32.1, OOD 평균 21.3으로 가장 낮아요. 특히 화각·차종이 다른 리그(R2 20.2, R6 9.4, R9 4.2)에서 처음 설정보다 낮아요.
- **depth 상태:** 독립 리뷰에서 epoch 5 EMA 체크포인트로 확인했어요.
  - UDGA-Ld는 depth를 제대로 배워요(최대 확률 0.43, LiDAR와 순위 상관 +0.95).
  - UDGA-Lp는 처음 설정처럼 균일하게 무너져 있어요(최대 확률 0.02, 상관 +0.04).
  - 세 설정 모두 학습 1회이고, R1 계열 차이(1~2점)는 한 체크포인트의 흔들림 범위 안일 수 있어요.
- **주의:** 세 UDGA 설정은 모두 예전 base(PCCR BEVDet 설정)에서 EMA가 아닌 가중치로 평가했어요. vanilla를 Megvii 공식 코드로 다시 학습하고 있어서(2026-11-23 결정), 그 결과를 보고 방법 행을 어떻게 할지 정해요.

## BEVDepth (Megvii 공식 코드, 공저자 base)

- **날짜:** 2026-11-23
- **코드:** `megvii_bevdepth/` = Megvii 공식 BEVDepth(@d78c7b5) + 공저자 PCCR 파일(perception_sota-bevdepth @7e8524f, 커밋 `e1e1996`). 공저자 plain exp는 `megvii_bevdepth/bevdepth/exps/pccr/bev_depth_lss_r50_pccr_24e.py`예요.
- **설정:** 단일 프레임, 256×704, 9개 클래스 1 task(회전 NMS 0.2), lr 2e-4 고정, AdamW 1e-7, MultiStepLR [19, 23], 24 epoch, GPU 2장 × 4, CBGS·EMA 없음, **FP32**예요.
  - 이 체크포인트는 제 첫 이식 exp(커밋 `6b6e123`, 지금은 지움)로 학습했어요. 공저자 plain exp와는 resize 하한(0.4825 vs 0.4826)만 달라요. info도 똑같아요([decisions.md](decisions.md) 2026-11-23).
- **체크포인트:** `megvii_bevdepth/outputs/pccr_bevdepth_r50/R1/lightning_logs/version_0/checkpoints/epoch=23-step=11328.ckpt`(마지막 epoch, EMA 아님)
- **채점:** 공저자 방식이에요. `-e --test_rigs`, GPU 1장, FP32, `megvii_bevdepth/pccr/eval_pccr_devkit.py`(nuscenes-devkit 1.1.10)예요.

  ```bash
  RUN=$PWD/megvii_bevdepth/outputs/pccr_bevdepth_r50/R1 bash tools/megvii_pccr_test_rigs.sh bev_depth_lss_r50_pccr_24e 0 \
      megvii_bevdepth/outputs/pccr_bevdepth_r50/R1/lightning_logs/version_0/checkpoints/epoch=23-step=11328.ckpt
  ```

- **결과(7-class mAP):** R1 32.1, OOD 평균 15.1, 최저 0.0(R5), 하락률 53.2%, H 20.5예요. 행은 위 두 표에 있어요.
  - R1 클래스별 AP: car 42.5, truck 34.6, bus 24.9, bicycle 45.9, motorcycle 39.1, adult 30.2, child 7.7이에요.
  - 예전 base vanilla(R1 30.4)보다 R1·R1-c6·R1-c10·R1-r가 1.5~2.5점 높아요. R1-t는 3.2점 낮고, R4는 같아요. 먼 리그는 둘 다 무너져요(R1-f 0.6, R2 0.6, R5 0.0).
  - 공저자 README에는 그쪽 plain 학습(amp)의 R6가 FP32 추론에서 7.43이라고 적혀 있어요. 우리 FP32 학습은 7.15라서 비슷해요.
  - 먼 리그가 무너지는 이유는 예전 base와 같아요. camera-aware MLP의 BatchNorm1d가 R1에서 상수인 intrinsic을 거의 0인 분산으로 나눠서, intrinsic이 다른 리그에서 입력이 수만 배가 돼요. 공저자 문서도 R5에서 BEVDepth 박스가 0이라고 적었어요.
- **원시 결과:** `.../pccr_test/epoch=23-step=11328/<RIG>/{results_nusc.json, eval.json}` (git 제외)
- **쓰지 않은 채점:** 학습 직후 자동으로 돈 제 이식의 `-p` 채점은 예측을 R1 val 길이(1020)로 잘라서 R1에서 멈췄어요. PCCR 채점기가 빠진 샘플을 잡아서 잘못된 수치는 나오지 않았어요.

## BEVDepth + PD-BEV-A / -B (Megvii base)

- **날짜:** 2026-11-23
- **코드:** `megvii_bevdepth/bevdepth/exps/pccr/pdbev_lss_r50_pccr_24e_{A,B}.py` (구성은 [decisions.md](decisions.md) 2026-11-23, 버그 수정 대응은 [bug_ledger.md](bug_ledger.md) "PD-BEV: Megvii base 이식")
- **학습:** FP32, GPU 2장 × 4, 24 epoch, 0.69 s/iter(약 2.2시간)예요. A는 GPU 2·3, B는 GPU 0·1이에요.

  ```bash
  bash tools/pdbev_megvii_train.sh A 2,3
  bash tools/pdbev_megvii_train.sh B 0,1
  ```

- **체크포인트:** `megvii_bevdepth/outputs/pdbev_lss_r50_pccr_24e_{A,B}/lightning_logs/version_0/checkpoints/epoch=23-step=11328.ckpt`(마지막 epoch, EMA 아님)
- **채점:** 공저자 방식, GPU 1장씩, FP32예요. 리그를 반씩 나눠 두 GPU에서 돌렸어요.

  ```bash
  RIGS="R1 R1-t R1-r R1-f R1-c6 R1-c10 R2" bash tools/megvii_pccr_test_rigs.sh pdbev_lss_r50_pccr_24e_A 2
  RIGS="R3 R4 R5 R6 R7 R8 R9" bash tools/megvii_pccr_test_rigs.sh pdbev_lss_r50_pccr_24e_A 3
  ```

- **결과(7-class mAP):**
  - PD-BEV-A: R1 33.6, OOD 평균 23.4, 최저 6.0(R5), 하락률 30.5%, H 27.6이에요.
  - PD-BEV-B: R1 34.0, OOD 평균 23.4, 최저 5.6(R5), 하락률 31.2%, H 27.7이에요.
  - R1 클래스별 AP(A / B): car 45.2 / 45.4, truck 36.2 / 36.5, bus 27.7 / 28.8, bicycle 45.5 / 46.5, motorcycle 44.9 / 44.6, adult 29.9 / 29.6, child 6.1 / 6.5예요.
- **관찰:**
  - 같은 base의 vanilla(R1 32.1, OOD 15.1)보다 14개 리그 모두 높아요. 화각·차량이 바뀐 리그가 크게 올라요(R1-f 0.6 → 19.9, R2 0.6 → 23.3, R3 10.4 → 25.4).
  - 예전 base와 달리 R1도 올라요(vanilla 대비 A +1.5, B +1.9). 예전 base에서는 PD-BEV가 R1을 2.6점 깎았어요.
  - 새 base에서는 A와 B의 OOD 평균이 같아요(23.385 vs 23.380). B는 R1-t가 1.8점 높고 R6가 0.9점 낮아요. 예전 base에서는 B가 OOD 1.5점 높았어요.
  - 학습은 1회씩이라, A·B 차이(1점 안팎)는 흔들림 범위 안일 수 있어요.

## BEVDepth + UDGA-Lp / -Ld (Megvii base)

- **날짜:** 2026-11-24
- **코드:** `megvii_bevdepth/bevdepth/exps/pccr/udga_lss_r50_pccr_24e_{Ld,Lp}.py` (구성은 [decisions.md](decisions.md) 2026-11-24, 버그 수정 대응은 [bug_ledger.md](bug_ledger.md) "UDGA: Megvii base 이식")
- **설정:**
  - UDGA-Ld = 공개 config 구성이에요: LiDAR depth(Megvii base 손실 ×3, 타깃 d·k) + L_ov + 공개 코드 L_p + pc400 + ext aug.
  - UDGA-Lp = 논문 최종 행이에요: L_ov + 논문 식 4의 L_p, LiDAR depth 없음, pc400 + ext aug.
  - 둘 다 24 epoch, FP32, GPU 2장 × 4예요.
- **학습:** Ld는 GPU 2·3에서 13:22~14:57에 돌았어요(0.48~0.52 s/iter, 약 1.6시간). Lp는 GPU 0·1이 비자 13:37~15:35에 돌았어요(0.6 s/iter, 약 2시간). 논문 식 4의 L_p는 미분 가능한 warping을 해서 Ld보다 느려요.

  ```bash
  bash tools/udga_megvii_train.sh Ld 2,3
  bash tools/udga_megvii_train.sh Lp 0,1
  ```

- **체크포인트:** `megvii_bevdepth/outputs/udga_lss_r50_pccr_24e_{Ld,Lp}/lightning_logs/version_0/checkpoints/epoch=23-step=11328.ckpt`(마지막 epoch, EMA 아님)
- **채점:** 공저자 방식, GPU 1장씩, FP32예요. PD-BEV처럼 리그를 반씩 나눠 두 GPU에서 돌렸어요. 리그마다 test 샘플 1800개를 모두 채점했어요.

  ```bash
  RIGS="R1 R1-t R1-r R1-f R1-c6 R1-c10 R2" bash tools/megvii_pccr_test_rigs.sh udga_lss_r50_pccr_24e_Ld 2
  RIGS="R3 R4 R5 R6 R7 R8 R9" bash tools/megvii_pccr_test_rigs.sh udga_lss_r50_pccr_24e_Ld 3
  ```

  Lp는 같은 방식으로 GPU 0·1에서 채점했어요.

- **검사:** `tools/check_udga_megvii.py`(CPU) 7개 모두 PASS, `tools/udga_megvii_smoke.py`(GPU) PASS예요.
- **결과(7-class mAP):**
  - UDGA-Lp: R1 35.0, OOD 평균 22.2, 최저 4.7(R9), 하락률 36.5%, H 27.2예요.
  - UDGA-Ld: R1 32.7, OOD 평균 20.2, 최저 2.8(R9), 하락률 38.2%, H 25.0이에요.
  - R1 클래스별 AP(Lp / Ld): car 46.9 / 43.0, truck 43.7 / 37.9, bus 29.0 / 27.7, bicycle 47.6 / 48.7, motorcycle 42.7 / 37.3, adult 30.8 / 27.9, child 4.4 / 6.6이에요.
- **관찰:**
  - Lp가 14개 리그 모두에서 Ld보다 높아요(R1 +2.3, OOD 평균 +2.0). 예전 base에서도 Lp가 Ld보다 높았어요.
  - 둘 다 같은 base의 vanilla(R1 32.1, OOD 15.1)보다 높아요. Lp는 14개 리그 모두 높고, Ld는 12개가 높아요(R6 7.08 vs 7.15, R9 2.81 vs 3.18만 조금 낮아요). 화각·차량이 바뀐 리그가 크게 올라요(Lp: R1-f 0.6 → 20.8, R2 0.6 → 21.3, R3 10.4 → 22.0).
  - 같은 base의 PD-BEV와 비교하면, Lp는 R1이 가장 높지만(35.0, PD-BEV-A 33.6, B 34.0) OOD 평균은 1.2점 낮아요(22.2 vs 23.4). H도 조금 낮아요(27.2 vs 27.6·27.7). PD-BEV-A보다 R1 계열 리그와 R4·R8에서는 높고, R2·R3·R5·R6·R7·R9에서는 1~5점 낮아요.
  - Ld는 PD-BEV-A보다 R1이 0.9점, OOD 평균이 3.2점 낮아요. R2·R3·R6·R7·R9에서는 6~8점 차이가 나요.
- **예전 base와 비교:**
  - 예전 base에서는 UDGA-Lp의 OOD 평균이 25.2로 방법 행 중 가장 높았어요. 새 base에서는 22.2로 2.9점 낮아져서 PD-BEV(예전 18.8·20.3 → 새 23.4)와 순서가 바뀌었어요.
  - 낮아진 곳은 초점거리가 R1과 다른 카메라가 있는 8개 리그(R1-f, R2, R3, R5, R6, R7, R8, R9)예요. 이 리그들에서 Lp는 2.0~7.2점(평균 4.7점) 낮아요. 초점거리가 R1과 같은 6개 리그(R1, R1-c6, R1-c10, R1-r, R1-t, R4)는 −1.3~+1.0점이에요.
  - 다른 행도 같은 모양이에요. 예전 base → 새 base 변화의 평균을 초점거리가 같은 리그 / 다른 리그로 나누면 Ld +0.5 / −2.1, PD-BEV-A +6.1 / +3.6, PD-BEV-B +5.4 / +1.8이에요. 모든 방법에서 초점거리가 다른 리그가 2.5~4.8점 덜 올라요.
  - **원인은 아직 몰라요.** camera-aware MLP의 BatchNorm 붕괴로는 두 base의 차이를 설명할 수 없어요. 두 base 모두 camera-aware MLP가 켜져 있고, 확인한 체크포인트 8개(예전 base vanilla·PD-BEV-A·UDGA-Ld·Lp, 새 base 같은 네 행)가 모두 같은 값을 가져요(fx·fy 분산 7.2e-10, scale γ 1.0). 이 값이면 초점거리가 다른 카메라에서 MLP 입력이 −5×10⁴(fx 834)에서 −2×10⁵(fx 370)까지 튀어요. 하지만 두 base에서 똑같이 일어나요.
- **depth 분포 (R1 test 8개 샘플, `tools/udga_megvii_depth_stats.py`):**
  - Lp는 예전 base처럼 거의 평평해요. 칸마다 softmax 최대 확률의 평균이 0.026이에요(89개 bin 균일 분포면 0.011). 기대 depth와 LiDAR depth의 순위 상관은 −0.67이에요.
  - Ld는 depth를 제대로 배워요. 최대 확률 0.65, 순위 상관 +0.93이에요.
  - 학습 끝의 L_ov도 Lp 0.011, Ld 0.10~0.13이에요. L_ov는 logit이 모두 같으면 0이 돼요.
