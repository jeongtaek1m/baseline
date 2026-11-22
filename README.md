# Gen2E — PCCR cross-rig baseline 포팅

Gen2E 논문 메인 표의 빈 baseline 행을 채우기 위한 코드 저장소예요. 표는 PCCR 벤치마크에서 **R1로 학습하고 14개 리그의 test split으로 평가**한 결과예요.

> **공저자끼리 코드를 교차 검토하기 위한 private 레포예요. 외부에 공개하거나 배포하지 않아요.**
> UDGA는 Samsung 소유 라이선스이고, PD-BEV와 PCCR 코드에는 라이선스가 없어요. PCCR 데이터는 CC-BY-NC-4.0이에요. 자세한 내용은 [라이선스 주의](#라이선스-주의)를 보세요.

## 채울 행과 진행 현황

| 표의 행 | View transform | Base 코드 | 상태 |
|---|---|---|---|
| BEVDepth (vanilla) | Forward | `bevdepth/` (PD-BEV 트리에 원본 BEVDet 클래스 복원) | 준비 중 |
| BEVDepth + PD-BEV | Forward | `bevdepth/` | 준비 중 |
| BEVDepth + UDGA | Forward | `bevdepth/` (UDGA를 mmdet3d 1.0 기준으로 재구현) | 준비 중 |
| BEVFormer-S (기존 행 재현 확인) | Backward | `bevformer/` (small 구조, 단일 프레임) | 준비 중 |
| BEVFormer-S + PD-BEV | Backward | `bevformer/` | 준비 중 |
| + UniDrive, + CoIn3D | — | — | 이번 범위에서 제외 |

진행 기록은 [docs/progress.md](docs/progress.md)에, 결정 사항과 근거는 [docs/decisions.md](docs/decisions.md)에 있어요.

## 레포 구조

```
Gen2E/
├── README.md
├── docs/                    결정 기록 · 평가 프로토콜 · 데이터 · 환경 · 포팅 계획 · 진행 기록
├── bevdepth/                BEVDepth 계열 작업 공간 (= PD-BEV 레포 @c2995d1, BEVDet dev2.0 기반)
├── bevformer/               BEVFormer-S 작업 공간 (= 공식 BEVFormer @66b65f3)
├── third_party/
│   ├── UDGA/                UDGA 원본 (@9198218, 참고용, 수정 금지)
│   └── pccr/                PCCR 벤치마크 원본 (@6798206: 데이터셋 어댑터, 설정, 공식 결과)
├── tools/
│   ├── download_pccr.sh     PCCR 다운로드 (HF 리비전 고정 + sha256 검증)
│   ├── pccr_sha256.txt
│   ├── pccr_rig_summary.py  리그 정의 요약표
│   ├── pccr_metric.py       표 지표(7-class mAP) 재현과 집계
│   ├── pccr_box_z_check.py  GT 박스 z 확인 (좌표계, 지면 접촉, 이미지 투영)
│   ├── setup_env_bevdepth.sh
│   └── setup_env_bevformer.sh
└── data/pccr -> /data2/PCCR (심볼릭 링크, git 제외)
```

### 커밋 규칙 (리뷰용)

공저자가 "원본 대비 우리가 바꾼 것"을 정확히 볼 수 있도록 커밋을 나눠요.

1. **첫 커밋:** upstream 원본 스냅샷만 넣어요. 수정 없이 넣고, 중첩 `.git`은 제거하고, 출처와 커밋은 아래 표에 기록해요.
2. **이후 커밋:** 한 커밋에 한 가지 변경만 담아요. 예: `setup.py` 복원 → PCCR 데이터셋 어댑터 → vanilla BEVDepth 원본 클래스 복원 → PD-BEV 설정 → UDGA 이식.
3. **변경 확인:** `git log --oneline -- bevdepth/`로 이력을 보고, `git diff <첫 커밋> -- bevdepth/`로 원본 대비 전체 변경을 봐요.
4. **PD-BEV 자체의 변경 확인:** PD-BEV가 원본 BEVDet에서 바꾼 부분(16개 파일 수정, 18개 추가)은 BEVDet dev2.0 @2ae6a02와 비교하면 돼요. 목록은 [docs/porting_plan.md](docs/porting_plan.md)에 있어요.

## 빠른 시작

```bash
# 1) 데이터 (약 161GB, /data2/PCCR/<RIG>/ 에 풀림)
bash tools/download_pccr.sh /data2/PCCR
mkdir -p data && ln -s /data2/PCCR data/pccr

# 2) 환경 (RTX 4090 기준, 버전은 docs/environment.md 참고)
bash tools/setup_env_bevdepth.sh      # gen2e-bevdepth
bash tools/setup_env_bevformer.sh     # gen2e-bevformer

# 3) 표 지표가 7-class mAP인지 직접 확인
python3 tools/pccr_metric.py verify-dagger
```

## 평가 프로토콜 요약

- **학습:** R1 train split, 7개 클래스(traffic 2종 제외). 모델당 GPU 2장.
- **평가:** 14개 리그 각각의 **test** split.
- **지표:** **7-class mAP.** traffic_sign과 traffic_light를 뺀 7개 클래스의 nuScenes 방식 AP 평균(거리 임계값 0.5/1/2/4 m)이에요. 표의 † 행 4개가 이 값과 소수점까지 일치해요.
- 자세한 내용은 [docs/evaluation_protocol.md](docs/evaluation_protocol.md)에 있어요.

## 문서

| 문서 | 내용 |
|---|---|
| [docs/decisions.md](docs/decisions.md) | 결정 사항 기록 (날짜, 근거, 확인이 필요한 항목) |
| [docs/evaluation_protocol.md](docs/evaluation_protocol.md) | 표 지표 검증, 학습·평가 규칙 |
| [docs/pccr_dataset.md](docs/pccr_dataset.md) | PCCR 데이터 구조, 14개 리그 정의, 주의점 |
| [docs/environment.md](docs/environment.md) | conda env 2개 구성과 4090 관련 메모 |
| [docs/porting_plan.md](docs/porting_plan.md) | baseline별 포팅 계획, 코드 분석에서 찾은 버그·하드코딩 |
| [docs/progress.md](docs/progress.md) | 날짜별 진행 기록 |

## Upstream 출처

| 경로 | 원본 | 커밋 | 라이선스 |
|---|---|---|---|
| `bevdepth/` | [EnVision-Research/Generalizable-BEV](https://github.com/EnVision-Research/Generalizable-BEV) (PD-BEV, AAAI 2025) | `c2995d1` | 없음 (BEVDet·mmdet3d의 Apache-2.0 코드 포함) |
| `bevformer/` | [fundamentalvision/BEVFormer](https://github.com/fundamentalvision/BEVFormer) | `66b65f3` | Apache-2.0 |
| `third_party/UDGA/` | [SAITPublic/UDGA](https://github.com/SAITPublic/UDGA) (NeurIPS 2024) | `9198218` | Samsung proprietary |
| `third_party/pccr/` | [BaderTim/plentiful-carla-camera-rigs](https://github.com/BaderTim/plentiful-carla-camera-rigs) (ECCV 2026) | `6798206` | 없음 |
| PCCR 데이터 | [timb2001/plentiful-carla-camera-rigs](https://huggingface.co/datasets/timb2001/plentiful-carla-camera-rigs) | `18edf4f` | CC-BY-NC-4.0 |
| (비교 기준) PD-BEV의 원 base | [HuangJunJie2017/BEVDet](https://github.com/HuangJunJie2017/BEVDet) dev2.0 | `2ae6a02` | Apache-2.0 |
| (설치 시 생성) | open-mmlab/mmdetection3d `v0.17.1` | — | Apache-2.0 |

## 라이선스 주의

- **UDGA:** Samsung Electronics 소유예요. 서면 허가 없이 복제, 배포, 공개할 수 없고, 비상업 학술 연구 목적으로만 쓸 수 있어요. UDGA 코드와 그 파생 코드(포팅 결과 포함)는 이 private 레포 밖으로 내보내지 않아요.
- **PD-BEV, PCCR 코드:** LICENSE 파일이 없어서 재배포 권한이 불명확해요.
- **PCCR 데이터:** CC-BY-NC-4.0이에요. 데이터 자체는 레포에 넣지 않아요.
- **나중에 공개하는 경우:** 위 코드를 빼고 Apache-2.0 부분만 남기는 별도 정리가 필요해요.
