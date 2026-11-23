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

- **BEVFormer-S 설정 결정 (사용자):** 배치 8(누적 4회)·lr 2e-4, 1280×720, ImageNet 초기화, CAN bus 끔이에요.
- **좌표계 결정 변경:** 사용자가 "LiDAR로 하면 차종마다 달라져서 ego 기준으로"라고 했어요. 그래서 원점을 R1 LiDAR 장착 위치로 고정한 차량 고정 좌표계로 바꿨어요([decisions.md](decisions.md)).
- **다른 cross-domain 연구 조사 (사용자 요청, "123d" 포함):** [pccr_dataset.md](pccr_dataset.md) "다른 cross-domain 연구의 좌표계"에 정리했어요.
  - 123D는 "123D: Unifying Multi-Modal Autonomous Driving Data at Scale"(arXiv 2605.08084)예요. §4.3에서 PETR과 BEVFormer-S로 cross-dataset 검출을 해요.
  - 좌표계를 명시적으로 다루는 연구(ST3D, Zheng 외 CoRL'23, PD-BEV)는 원점을 지면 쪽으로 맞춰요.
- **BEVFormer-S 포팅 (0-C, 4장):** 커밋 `99f5fd0`~`7df3601`이에요. 구현과 확인 결과는 [porting_plan.md](porting_plan.md) 0-C의 "구현" 단락에 있어요.
  - gen2e-bevformer env의 Pillow를 9.5.0으로 고정했어요. detectron2 0.6과 Pillow 10이 충돌해서 플러그인 import가 실패했어요.
  - 스모크 테스트(R1 mini, 1 epoch)에서 학습은 0.72 s/iter, GPU당 11.5 GB로 돌았어요. loss는 23.0에서 18.8로 내려갔어요.
  - 스모크 체크포인트로 R5(카메라 4대)와 R1-c10(10대) test 평가 경로도 확인했어요.
- **BEVFormer-S R1 학습을 시작했어요** (20:42, GPU 2·3). 0.77 s/iter이고 약 10시간 걸려요. 출력은 `bevformer/work_dirs/bevformer_small_pccr/R1/`, 로그는 `logs/train_bevformer_small_pccr_R1.log`예요.

- **결과 기록 도구 보강 (리뷰 반영):** 리그 평가 결과마다 체크포인트를 남기고, 집계에서 결과가 없는 리그는 `--`로 표시해요. 스크립트의 PORT 인자는 숫자일 때만 PORT로 읽어요.
- **PD-BEV·UDGA 결정 (사용자)**
  - 공식 코드 버그는 논문 의도대로 고쳐요.
  - car 전용 BEV 보조 손실은 빼요. 논문에 없고, 공식 학습에서도 동작하지 않았어요.
  - 손실 정규화는 공식 그대로 둬요.
  - 기본 구조는 vanilla BEVDepth와 같게 해요.
  - 섭동 범위는 A(논문·공식)와 B(PCCR 리그) 두 가지를 다 돌려요.
- **버그 장부 [bug_ledger.md](bug_ledger.md):**
  - 버그마다 발견 경위를 적었어요: 2026-11-21 분석 → 독립 검증 → 2026-11-22 재확인. 원자료 JSON은 `docs/analysis/2026-11-21/`에 넣었어요.
  - 재현 체크(`tools/bugrepro/pdbev_checks.py`)는 공식 코드에서 11개가 모두 FAIL이었어요. 버그를 하나씩 고쳐서 모두 PASS가 됐어요.
  - 구현하면서 새로 찾은 문제도 넣었어요: 보조 head의 ReLU 바닥(PDBEV-17), 평가 크래시(PDBEV-18).
- **PD-BEV 포팅:** 커밋 `0318e03`~`a1ee9cd`예요.
  - PCCR 카메라 적응, 버그 수정 16개, BEV 보조 손실 제거, 설정 A·B를 넣었어요.
  - GPU 스모크에서 BEV_aux가 step당 1번 돌고, 추론이 되고, 그래디언트가 462/462개 파라미터에 흘렀어요.
  - 2-GPU mini 학습과 평가도 통과했어요(0.91 s/iter, GPU당 13.7 GB).
- **PD-BEV-A R1 학습을 시작했어요** (23:13, GPU 0·1). 약 3시간 반 걸려요. 출력은 `bevdepth/work_dirs/pdbev-r50-pccr-A/R1/`예요.

## 2026-11-23

- **학습 종료 → 평가 시작 (09:15):** BEVFormer-S(06:48)와 PD-BEV-A(02:33)가 24 epoch을 끝냈어요. 14개 리그 평가를 GPU 0·1(PD-BEV-A)과 2·3(BEVFormer-S)에서 돌렸어요. PD-BEV-B 학습은 A 평가 뒤에 GPU 0·1에서 자동으로 시작했어요.
- **PD-BEV-A 결과:** 7-class mAP가 R1 27.8이에요. vanilla(30.4)보다 2.6점 낮지만 화각이 다른 리그에서 크게 올랐어요(R1-f 0.2 → 15.3, R2 0.1 → 18.7, R3 5.7 → 20.8). 행 전체는 [results.md](results.md)에 있어요.
- **사용자 표와 비교 (사용자 요청):** 사용자 표의 BEVDepth [42]는 R1 36.9로, 우리 vanilla보다 6.4점 높아요. R1 계열 리그는 모두 5.5~6.9점 차이이고, 먼 리그는 비슷해요(하락률 53.5% vs 55.0%). 표의 열 정의(OOD 평균 = R1 제외 13개 평균, 최저, 하락률, 14개 평균, R1과 OOD 평균의 조화평균)를 표 값으로 재현해서, 같은 형식의 우리 행을 드렸어요. BEVDepth 행을 바꿀지는 사용자가 직접 다시 검증한 뒤 정해요.
- **BEVDepth 좌표계·평가 검증 (사용자 질문 "우리 config랑 좌표계 잘 실험했지?"):** `tools/check_bevdepth_pccr.py`를 만들었어요(커밋 `627ed75`).
  - `eval-gt`: 학습 GT를 예측으로 넣으면 R1·R6에서 mAP 0.998, ATE 0.002 m, ASE 0, AOE 0.0004예요.
    - bus 0.989는 devkit의 recall 1.0 반올림 문제예요(모든 거리 기준에서 1 − 1/90).
    - ATE는 PCCR 어댑터가 바닥 중심(`box[:3]`)을 쓰는 데서 와요. ego가 기운 샘플(test의 1.3%가 5° 초과, 최대 11°)에서 최대 0.41 m예요. vanilla R1 예측을 중력 중심으로 다시 평가해도 mAP 차이는 +0.02예요.
    - 처음에는 점수를 모두 1로 줘서 devkit이 오차를 박스 하나 값으로 보고했어요(ATE 0.156). 점수를 서로 다르게 줘서 고쳤어요.
  - `render`: 이미지 증강과 BDA가 걸린 학습 샘플에서 박스와 LiDAR depth 라벨이 이미지와 맞아요.
- **UDGA 결정 (사용자):** LiDAR depth 없음, pc400 + bin [1, 90], ext aug 포함(BDA, 확률 0.5), 공간 쌍만, L_ov는 대응 칸만, L_p는 일단 공개 코드대로예요([decisions.md](decisions.md)). 논문(arXiv 2410.22461)은 표 4와 부록 B를 직접 읽었어요.
- **UDGA 재구현:** 커밋 `c1982eb`~`31ecc5f`예요. 새 파일 3개와 설정이에요. 기존 파일은 고치지 않았어요.
  - 재현 체크(`tools/bugrepro/udga_checks.py`)는 공식 함수를 소스 그대로 꺼내 돌려요. 공식은 9개 모두 FAIL이에요. 재구현은 7개 PASS이고, 2개는 결정대로 n/a예요(UDGA-2 LiDAR 타깃 없음, UDGA-4 공개 코드 유지).
  - R1(8대)에서 쌍은 이웃끼리 16개예요. nus6 합성 링에서는 공식 12쌍과 같아요.
  - GPU 스모크와 2-GPU 1 epoch mini 학습(평가 포함)이 통과했어요.
- **장부:** UDGA 부분을 PD-BEV와 같은 형식(위치·문제·발견·재확인·재구현·커밋·체크 전/후)으로 채웠어요. PD-BEV 체크 표에서 칸이 밀린 PDBEV-9 행도 고쳤어요.

- **BEVFormer-S 결과:** 7-class mAP가 R1 34.5, OOD 평균 25.1, 하락률 27.3%예요. 화각이 바뀐 리그에서도 거의 무너지지 않아요(R1-f 24.7, R2 23.4). 행 전체와 검증 내용은 [results.md](results.md)에 있어요.
- **BEVFormer-S 독립 검증 워크플로:** 기존 표 행(R1 21.1)과 차이가 커서 돌렸어요. 관점은 넷이에요: 출처(리그 데이터·체크포인트·split), mAP 재계산, 누수·정상성, 좌표계. 에이전트 5개가 37분 동안 CPU로 확인했어요. 판정은 "값은 신뢰 가능, 비교 시 주의점 있음"이에요. 원자료는 `docs/analysis/2026-11-23/verify_bevformer_row.json`이에요.
  - 주의점: 점수 하한 차이(BEVFormer는 하한 없음, BEVDepth는 0.1; 맞추면 R1 −1.3), 다른 차량 리그 5개에서 †행과 좌표계가 다름, 학습 1회예요.
  - 검증이 지적한 재현성 공백을 메웠어요. 평가 스크립트가 리그별 data_root와 ann_file을 로그와 `data.txt`에 남겨요(커밋 `54e83fa`).
- **UDGA depth 붕괴 관찰:** epoch 6에서 `loss_overlap`이 0.27에서 0.012로 떨어졌어요. epoch 4 체크포인트로 R1 val 이미지의 depth 분포를 CPU에서 재 보니 거의 균일했어요.
  - UDGA: 최대 확률 0.022(균일 0.011), 엔트로피 4.45(균일 4.49), logit 크기 0.12예요.
  - vanilla: 최대 확률 0.47, 엔트로피 1.8, logit 크기 3.9예요(같은 epoch 4).
  - L_ov는 모든 logit이 같으면 0이 되는데, LiDAR depth도 없고 L_p 그래디언트도 없으니 모델이 그쪽으로 가요. 그래도 val mAP는 LiDAR depth가 없는 PD-BEV-A와 비슷하게 올라요(epoch 6: 12.3 vs 13.4). 어떻게 할지 사용자에게 물었어요.

- **Megvii base (저녁):** 공저자가 perception_sota-bevdepth @7e8524f에 Megvii 공식 BEVDepth용 PCCR 파일을 올렸어요. 사용자 결정에 따라 제 이식을 공저자 파일로 바꾸고(커밋 `e1e1996`, 제 파일 삭제 `aad3a38`), 그 위에 PD-BEV-A·B를 FP32로 옮겼어요([decisions.md](decisions.md) 2026-11-23).
  - 공저자 info를 우리 데이터 경로로 다시 만들었어요(`megvii_bevdepth/data/pccr_megvii/`, 15개). 제 info와 토큰 순서·카메라 순서·주석·속도·점 수가 같아요.
  - **vanilla(Megvii FP32 체크포인트, 공저자 채점):** R1 32.1, OOD 평균 15.1, 하락률 53.2%예요([results.md](results.md)). 학습 직후 돈 제 이식의 `-p` 채점은 예측을 R1 val 길이(1020)로 잘라서 쓰지 않았어요.
  - **PD-BEV 이식:** 새 파일 7개예요(커밋 `605bc09`, `0043e7a`, `9f10c0f`). 예전 base의 수정(PDBEV-1~18)을 그대로 옮겼어요([bug_ledger.md](bug_ledger.md) "PD-BEV: Megvii base 이식").
  - **검사:** `tools/check_pdbev_megvii.py` 6개 모두 PASS예요. 2D 타깃은 독립 투영과 peak가 같고(86개), 타깃 중심을 같은 카메라의 기하로 되돌리면 박스 중심과 1e-05 m 안에서 맞아요. GPU 스모크도 PASS예요(파라미터 477/477개 그래디언트, 배치 4에서 17.8 GB).
  - **학습:** PD-BEV-A는 GPU 2·3(20:30), PD-BEV-B는 GPU 0·1(20:53)에서 돌고 있어요. 0.7 s/iter, 각각 약 2.2시간이에요.
  - **공식 Megvii 코드에서 본 것:** CenterHead heatmap bias(−2.19)가 적용되지 않아서 첫 step 검출 손실이 약 2.5e3이에요(vanilla, PD-BEV 같음). 공식 그대로 둬요.
- **GitHub:** `jeongtaek1m/baseline`은 아직 공개 상태예요(2026-11-23 20:55 확인). UDGA 코드(Samsung 소유)가 들어 있어요. 로컬 main은 그 뒤로 커밋이 더 있고, 푸시하지 않았어요.

## 다음 할 일

1. PD-BEV-A(약 22:45)와 PD-BEV-B(약 23:05)가 끝나면 14개 리그를 채점해요. GPU 1장씩 쓰고, 리그를 반씩 나눠 돌릴 수 있어요: `RIGS="R1 R1-t R1-r R1-f R1-c6 R1-c10 R2" bash tools/megvii_pccr_test_rigs.sh pdbev_lss_r50_pccr_24e_A 2`
2. 결과를 [results.md](results.md)의 두 표에 넣고 vanilla(Megvii)와 비교해요.
3. 사용자 확인이 필요한 것: 새 base 구성(제 판단 두 가지, [decisions.md](decisions.md) 2026-11-23), UDGA를 새 base로 옮길지, 예전 base 행을 EMA로 다시 평가할지, 표에 넣을 BEVDepth 행.
4. BEVFormer-S + PD-BEV를 만들어요([porting_plan.md](porting_plan.md) 5장).
