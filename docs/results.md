# 결과

모든 수치는 **7-class mAP(%)**예요. R1 train으로 학습하고, 각 리그의 **test** split으로 평가했어요. 지표 정의는 [evaluation_protocol.md](evaluation_protocol.md)에 있어요.

| 행 | R1 | R1-c6 | R1-c10 | R1-f | R1-r | R1-t | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **BEVDepth (vanilla, 우리)** | 30.4 | 27.8 | 33.9 | 0.2 | 30.8 | 29.1 | 0.1 | 5.7 | 31.1 | 0.0 | 6.2 | 9.2 | 5.5 | 4.1 |
| (참고) BEVDet† | 25.2 | 23.2 | 27.1 | 0.5 | 24.7 | 21.7 | 1.3 | 9.0 | 25.5 | 0.0 | 8.2 | 11.2 | 4.9 | 5.1 |
| (참고) BEVFusion-C† | 26.0 | 24.0 | 27.1 | 0.9 | 24.0 | 21.7 | 2.2 | 14.6 | 25.7 | 0.0 | 10.6 | 12.8 | 3.6 | 6.5 |
| (참고) Fast-BEV† | 12.9 | 12.5 | 19.2 | 6.0 | 17.6 | 18.3 | 7.0 | 9.0 | 6.1 | 1.2 | 0.0 | 9.0 | 1.8 | 0.0 |
| (참고) PETR† | 21.1 | 18.8 | 21.0 | 2.3 | 18.2 | 13.4 | 1.8 | 10.7 | 17.0 | 0.1 | 0.0 | 5.8 | 1.8 | 0.0 |

† 행은 PCCR 공식 결과에서 다시 계산한 값이에요(`python3 tools/pccr_metric.py verify-dagger`). 9개 클래스로 학습한 모델이에요.

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
