# 결과

모든 수치는 **7-class mAP(%)**예요. R1 train으로 학습하고, 각 리그의 **test** split으로 평가했어요. 지표 정의는 [evaluation_protocol.md](evaluation_protocol.md)에 있어요.

| 행 | R1 | R1-c6 | R1-c10 | R1-f | R1-r | R1-t | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **BEVDepth (vanilla, 우리)** | 30.4 | 27.8 | 33.9 | 0.2 | 30.8 | 29.1 | 0.1 | 5.7 | 31.1 | 0.0 | 6.2 | 9.2 | 5.5 | 4.1 |
| **BEVDepth + PD-BEV-A (우리)** | 27.8 | 25.1 | 29.6 | 15.3 | 27.5 | 27.4 | 18.7 | 20.8 | 28.5 | 4.3 | 11.9 | 16.4 | 11.4 | 7.3 |
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
