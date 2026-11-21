# 평가 프로토콜

## 1. 표의 지표는 7-class mAP예요

Gen2E 메인 표에서 †가 붙은 4개 행(BEVDet, BEVFusion-C, Fast-BEV, PETR)은 PCCR 공식 결과에서 왔지만, PCCR이 출력하는 **9-class mAP가 아니에요.**
9개 클래스 중 `traffic_sign`과 `traffic_light`를 뺀 **7개 클래스**(car, truck, bus, bicycle, motorcycle, adult, child)의 AP 평균이에요.

- **계산법:** 각 클래스 AP는 nuScenes 방식 center-distance AP를 임계값 0.5/1/2/4 m에서 평균한 값이에요. 7개 클래스의 AP를 평균하고 100을 곱해요.
- **데이터:** R1에서 학습하고, 각 리그의 `<RIG>_infos_test.pkl`(test split)로 평가해요. PCCR `sequential_test.sh`의 기본값이 `EVAL_SPLIT=test`예요.
- **검증:** `third_party/pccr/models/results/<Model>/trained_on_R1.json`에 있는 클래스별 AP로 다시 계산하면 4개 모델 × 14개 리그가 모두 표와 0.05 이내로 일치해요(반올림 오차).

```bash
python3 tools/pccr_metric.py verify-dagger
# BEVDet     max|7-class mAP - table| = 0.049
# BEVFusion  max|7-class mAP - table| = 0.047
# Fast-BEV   max|7-class mAP - table| = 0.047
# PETR       max|7-class mAP - table| = 0.047
# RESULT: all dagger rows reproduced (<= rounding)
```

참고로 PCCR 공식 9-class mAP는 BEVDet R1 기준 19.7이고, 7-class mAP는 25.2예요(표 값).

### 클래스별 평가 거리 (PCCR `class_range`)

| 클래스 | 거리 (m) |
|---|---|
| car, truck, bus | 50 |
| bicycle, motorcycle, adult, child | 40 |
| traffic_sign, traffic_light | 30 (7-class mAP에서는 제외) |

- **NDS:** PCCR 어댑터는 `discard_velocity_eval=True`라서 속도 오차(mAVE)를 빼고 NDS를 다시 계산해요. 표에는 NDS를 쓰지 않지만 함께 기록해 둬요.

## 2. 학습·평가 규칙

| 항목 | 규칙 | 출처 |
|---|---|---|
| 학습 데이터 | R1 `R1_infos_train.pkl` | PCCR `sequential_train.sh` |
| 평가 데이터 | 14개 리그 각각의 `<RIG>_infos_test.pkl` | PCCR `sequential_test.sh` (`EVAL_SPLIT=test`) |
| 체크포인트 | 학습 마지막 체크포인트 (`latest.pth`) | PCCR `sequential_test.sh` |
| GPU | 모델당 2장 (4090 24GB) | 사용자 지시, PCCR BEVDet도 2장 |
| 입력 해상도 | 원본 1280×720을 384×704로 (BEVDepth 계열) | PCCR `bevdet-r50-pccr.py` |
| 학습 클래스 | 9개 (PCCR 기본) → 보고는 7-class | 확인 필요 ([decisions.md](decisions.md)) |

### PCCR 공식 BEVDet 설정 (BEVDepth 계열의 기준점)

`third_party/pccr/models/BEVDet/bevdet-r50-pccr.py` 기준이에요.

- **모델:** ResNet-50, CustomFPN, `LSSViewTransformer`(grid 0.8 m, ±51.2 m, depth 1~60 m), CenterHead(9-class 단일 task).
- **학습:** AdamW lr 2e-4, weight decay 1e-7, grad clip 5, step 24, 24 epoch, `MEGVIIEMAHook`.
- **배치:** `samples_per_gpu=4` × GPU 2장 = 8. 스크립트 주석에 "GPU 수를 바꾸면 lr을 조정하라"고 적혀 있어요.
- **데이터 증강:** image resize (-0.06, 0.11), rot ±5.4°, flip / BEV rot ±22.5°, scale 0.95~1.05, flip.

### 우리 결과 집계

nuScenes devkit이 리그마다 `metrics_summary.json`을 만들면, 아래 명령으로 7-class mAP 표 행을 만들어요.

```bash
python3 tools/pccr_metric.py summarize 'work_dirs/<model>/R1/test_*_on_*_test/**/metrics_summary.json'
```

## 3. 주의점

- **카메라 자동 선택:** PCCR 공식 BEVDet 설정은 `data_config.cams=None`으로 리그마다 카메라를 자동으로 골라요. 그런데 이걸 처리하는 `PrepareImageInputs` 수정본은 PCCR 레포에 없어요. 원본 BEVDet은 `cams=None`이면 `len(None)`에서 실패해요. 우리가 직접 구현해야 하고, † 수치와 미세한 차이가 생길 수 있어요.
- **R5:** 카메라가 4대뿐이고 FOV가 120°예요. 거의 모든 방법에서 0점대가 나오는 리그예요. 평가가 정상적으로 도는지(크래시가 나지 않는지)만 따로 확인해야 해요.
