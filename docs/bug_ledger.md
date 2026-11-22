# 공식 코드 버그 기록 (PD-BEV, UDGA)

2026-11-22 결정에 따라 PD-BEV와 UDGA 공식 코드의 버그는 **논문 의도대로 고쳐요**([decisions.md](decisions.md)). 이 문서에는 버그마다 세 가지를 남겨요.
1. 어떻게 찾았는지
2. 무엇을 정확히 바꿨는지(전/후)
3. 고친 뒤 어떻게 확인했는지

"수정"과 "커밋" 칸은 고칠 때마다 채워요.

## 어떻게 찾았나

| 단계 | 날짜 | 방법 | 원자료 |
|---|---|---|---|
| 1. 코드 분석 | 2026-11-21 | 코드베이스마다 분석 에이전트가 코드를 읽고, 문제를 `파일:줄`과 함께 보고했어요(`bugs_or_gaps`). 대상은 PD-BEV, UDGA, BEVFormer, BEVDepth base, UniDrive/CoIn3D예요. | [analysis/2026-11-21/](analysis/2026-11-21/) `analysis_*.json` |
| 2. 교차 검증 | 2026-11-21 | 다른 에이전트가 보고된 위치를 독립적으로 다시 읽고 판정했어요(confirmed / partly_wrong / wrong / unverifiable). 분석이 놓친 문제도 추가했어요(`missed`). 일부는 실행으로 확인했어요. 예: UDGA 재투영 버그를 순수 Python 인덱스 재현으로 돌리니 D=3, h=2, w=4에서 24개 중 18개가 불일치했어요. | `verify_pdbev.json`, `verify_udga.json`, `verify_v1.json` |
| 3. 재확인 | 2026-11-22 | 고치기 전에 제가 해당 줄을 다시 읽었어요(아래 "재확인" 칸). 고칠 때는 재현 테스트를 먼저 써서 실패를 기록하고, 고친 뒤 같은 테스트가 통과하는지 확인해요. | 이 문서, `tools/bugrepro/` (고칠 때 추가) |

표의 "발견" 칸에 적힌 ID는 원자료 번호예요.
- 분석 결과: `analysis_*.json`의 `bugs_or_gaps[i]`는 PD-BEV `B<i>`, UDGA `U<i>`로 적었어요.
- 검증 판정: `verify_*.json`의 `checks[i]`는 PD-BEV `C<i>`, UDGA `V<i>`예요.
- 검증에서 추가로 찾은 것: `missed[i]`는 PD-BEV `M<i>`, UDGA `W<i>`예요.

## PD-BEV: 수정 대상 (DG 학습·평가 경로)

위치는 `bevdepth/mmdet3d/` 기준이에요(PD-BEV @c2995d1).

| ID | 위치 | 문제 | 발견 | 재확인 (2026-11-22) | 영향 | 수정 | 커밋 |
|---|---|---|---|---|---|---|---|
| PDBEV-1 | `models/necks/view_transformer_pc.py:387` | IFV에서 렌더링할 때 `bev[b, :, Z, X, Y]`로 읽어요. BEV 텐서는 `[B, C, Z, Y, X]`라서 x와 y가 뒤바뀌어요. | B08 → C05 confirmed | 코드에서 확인 | 렌더링된 PV feature가 x=y에 대칭인 셀을 읽어요. | `view_transformer_pc.py:387`에서 `bev[b, :, Z, X, Y]`를 `bev[b, :, Z, Y, X]`로 바꾸고, `:373`의 차원 이름도 `(B, C, Z, Y, X)`로 고쳤어요. |  |
| PDBEV-2 | `models/dense_heads/centerpoint_head.py:659` | `bev_loss`가 `pos_weights = gts.eq(0)`이에요. 배경 픽셀을 positive로 써요. head task 0이 1개 클래스일 때만 동작해요(`:656` `squeeze(1)`). | B09 → C06 confirmed, M01 | 코드에서 확인 | car BEV 보조 손실이 배경을 물체처럼 학습시켜요. 7클래스 단일 task head에서는 크래시가 나요. | **손실 자체를 빼요**(2026-11-22 결정, 논문에 없음). PDBEV-2 버그는 더 이상 해당하지 않아요. | |
| PDBEV-3 | `datasets/pipelines/loading_UDA.py:549-550` vs `:538-543` (사용: `models/detectors/pcbev.py:76-79`) | 가상 카메라 렌더링에는 `rot_aug`(key-ego 기준 절대 회전)와 `tran_aug = sensor2sensor_aug[:3, 3]`(상대 이동)을 짝지어 써요. 가상 뷰 타깃에는 절대 이동(t+δ)을 써요. | B11 → C08 partly_wrong (불일치는 맞고, 크기 설명만 일부 고침) | 코드에서 확인 | 렌더링한 카메라와 타깃 카메라의 위치가 달라요(수 m). | `loading_UDA.py:550`의 `tran_aug = sensor2sensor_aug[:3, 3]`(상대 이동)를 `sensor2keyego_aug[:3, 3]`(가상 카메라의 key-ego 기준 위치)로 바꿨어요. rot_aug와 짝이 맞고, 가상 뷰 타깃의 카메라 위치(t+δ)와 같아요. |  |
| PDBEV-4 | `datasets/pipelines/loading_UDA.py:929, 1063` | `gt_labels[i] < classes_num` 검사를 −1(학습하지 않는 클래스)도 통과해요. 그래서 음수 인덱스로 마지막 클래스 채널에 그려져요. | M03 (검증에서 추가) | 코드에서 확인 | PCCR 7클래스 설정에서는 traffic_sign/light가 child 채널에 그려져요. nuScenes에는 −1 라벨이 없어서 생기지 않던 문제예요. | 실제 뷰와 가상 뷰 두 곳(`loading_UDA.py:950, 1084`)의 라벨 검사를 `0 <= label < C`로 바꿨어요. |  |
| PDBEV-5 | `datasets/pipelines/loading_UDA.py:860, 998` | 2D 타깃을 범위·이름 필터를 거치기 전의 `results['ann_infos']`로 만들어요. | M04 | 코드에서 확인 | BEV 범위(±51.2 m) 밖이나 학습하지 않는 클래스의 물체도 2D·렌더 타깃이 돼요. | `Load3DBoxesHeatmap`에 `filter_objects`를 추가해서 두 뷰 모두 2D 타깃을 만들기 전에 거르게 했어요. 남기는 물체는 라벨 0..C−1이고, BEV 증강(`img_inputs[6]`) 뒤 중심이 BEV 범위 안인 것이에요(ObjectNameFilter·ObjectRangeFilter와 같은 기준). 범위는 새 인자 `point_cloud_range`로 받고, None이면 공식과 같아요. |  |
| PDBEV-6 | `datasets/pipelines/loading_UDA.py:938-952, 1072-1086` | 타깃을 `y_int-1:y_int+2` 슬라이스로 써요. crop 뒤 중심이 맵 위쪽 밖(`y_int ≤ −3`)이면 시작과 끝이 둘 다 음수라 **맵 아래쪽 행에** 써져요. | M02 | 코드에서 확인 | 엉뚱한 위치에 회귀·depth 타깃이 써져요. | | |
| PDBEV-7 | `datasets/pipelines/loading_UDA.py:868-876, 898` | 유효성 검사를 카메라 0의 이미지 크기와 **증강 전** 좌표로 해요. | B20 → C14 confirmed | 코드에서 확인 | PCCR은 카메라 크기가 모두 같아서 크기 부분은 무해해요. 증강 전 좌표로 검사하는 게 PDBEV-6의 원인이에요. | | |
| PDBEV-8 | `models/necks/view_transformer_pc.py:253` | IFV z-bin 크기를 `int(z범위 / height_num)`로 계산해요. | M05 | 코드에서 확인 | 공식 설정(8 m / 4칸)과 PCCR 설정(12 m / 4칸)은 나누어떨어져서 **발동하지 않아요.** 나누어떨어지지 않으면 z 범위 일부가 빠지거나, 0으로 나누게 돼요. | `view_transformer_pc.py:253`의 `int(z / height_num)`를 `float(z) / height_num`으로 바꿨어요. |  |
| PDBEV-9 | `datasets/pipelines/loading_UDA.py:591-595` | 실제 intrinsic(`intrins_actual`)에 resize와 crop만 반영하고 flip은 반영하지 않아요. | M13 | 코드에서 확인 | 뒤집힌 샘플은 DepthNet 카메라 입력의 cx'가 틀려요. | flip이면 실제 intrinsic의 주점을 `cx' = W_crop − cx'`로 옮겨요(`loading_UDA.py:595` 다음). 좌우 반전 자체는 공식대로 post_rot이 담당해요. 회전(±5.4°)도 공식대로 넣지 않아요. |  |
| PDBEV-10 | `datasets/pipelines/loading_UDA.py:351-381` | 섭동용 Euler 각 이름이 카메라 축과 맞지 않아요. `pitch_aug`는 영상을 화면 안에서 돌리고(roll), `roll_aug`는 카메라를 위아래로 기울여요(pitch). | M09 | 코드에서 확인 | PD-BEV-A는 두 범위가 같아서(±0.04 rad) 영향이 없어요. PD-BEV-B에서 pitch 범위를 따로 줄 때는 중요해요. | `euler_to_quaternion`에서 `pitch_aug`를 Euler roll(카메라 x축 회전 = 기울기)에, `roll_aug`를 Euler pitch(광축 회전)에 더하게 바꿨어요. yaw는 그대로예요. |  |
| PDBEV-11 | `models/detectors/pcbev.py:90-92` → `view_transformer_pc.py:312, 345, 353` | 같은 BEV에 `BEV_Aux`를 iteration마다 3번 돌려요. | M07 | 코드에서 확인 | BatchNorm running stat이 한 step에 3번 갱신돼요. 계산도 3배예요. | | |
| PDBEV-12 | `models/necks/img_aux.py:869-871` | IFV(높이 × feature)에 sigmoid가 없어요. 논문 식 3에는 있어요. | B22 → C22 confirmed | 코드에서 확인 | 논문과 계산이 달라요. | `BEV_Aux.forward`에서 `bev_height`에 `.sigmoid()`를 붙였어요(논문 식 3). PDBEV-17의 선형 출력층이 있어야 0~1 전 범위를 써요. |  |
| PDBEV-13 | `models/necks/img_aux.py:73-83` | 2D heatmap focal loss가 negative 항을 negative 픽셀 수로 나눠요. CenterNet은 positive 수로 나눠요. | M08 | 코드에서 확인 | 2D·렌더 heatmap 손실의 크기가 달라져요. 논문 표기를 확인한 뒤 고칠지 정해요. | **공식 유지**(2026-11-22 결정: 논문에 정규화 명시 없음) | |
| PDBEV-14 | `models/necks/img_aux.py:316-352, 640-676` | 회귀 손실을 전체 픽셀로 평균해요(positive는 몇 %뿐이에요). | B25 | 미확인 | 속성 손실의 실제 가중치가 아주 작아요. 논문 표기를 확인한 뒤 정해요. | **공식 유지**(2026-11-22 결정: 논문에 정규화 명시 없음) | |
| PDBEV-15 | `models/detectors/pcbev.py:62-69, 110-124` | `PCBEV_DG.simple_test`가 고장 나 있어요(시그니처 불일치, 학습 전용 입력 필요). | B07 → C03 confirmed | 코드에서 확인 | 공식 평가는 `BEVDepth_DG` 설정으로 해요. 우리도 추론은 기본 검출기 경로로 해요. | | |
| PDBEV-16 | `datasets/pipelines/loading_UDA.py:260` | 테스트 때 crop이 무작위예요. | B21 | 미확인 | 평가가 실행마다 달라질 수 있어요. | 테스트 crop을 원본 BEVDet 공식 `int((1 − mean(crop_h)) · newH) − fH`로 바꿨어요. 결정적이고, 학습 crop처럼 아래에 맞춰요. |  |
| PDBEV-17 | `models/necks/img_aux.py:190-198, 214-222` (Img_Aux, Img_Aux_Dy, BEV_Aux 모두) | 보조 head의 마지막 단이 `ConvTranspose → BN → ReLU`(또는 ReLU로 끝나는 블록)라서 출력이 항상 0 이상이에요. 그래서 heatmap = sigmoid(출력)이 **0.5 밑으로 못 내려가요.** IFV의 `sigmoid(F_height)`도 0.5~1에 갇혀요. | 2026-11-22 매핑 에이전트 (CPU에서 출력 범위 검사) | 코드에서 확인 (`img_aux.py:190-198, 405-414`) | 배경을 "물체 없음"으로 예측할 수 없어요. bev_loss는 eq(0) 버그와 겹쳐서 사실상 아무 일도 안 해요. | Img_Aux, Img_Aux_Dy, BEV_Aux 끝에 1×1 선형 conv(`out_conv`)를 더했어요. identity로 초기화해서 학습 시작 시점의 출력은 공식과 같아요. head 출력을 만드는 7곳(`feats = feats[-1]` 다음)에 적용했어요. |  |
| PDBEV-18 | `models/detectors/pcbev.py` + 평가 훅 | `PCBEV_DG`로 학습하면서 epoch마다 평가하면, 첫 평가에서 PDBEV-15 때문에 크래시가 나요. 공식 학습은 평가 간격(24)이 학습 epoch(23)보다 길어서 평가가 한 번도 안 돌았어요. | 2026-11-22 매핑 에이전트 | 코드 읽기 | 우리 설정(epoch마다 평가)에서는 1 epoch 뒤에 멈춰요. | PDBEV-15를 고치면 같이 해결돼요. | |

**논문과 코드가 다른 값:** yaw 섭동은 논문 부록 C.1 본문에서 0.2 rad, 공식 설정에서 0.4 rad예요(B22 → C22). 논문 그림 9와 표 1 수치는 0.4와 맞아서, 사용자 결정으로 PD-BEV-A는 0.4를 써요(2026-11-22).

**논문으로 확인한 의도 (arXiv 2310.11346v3, 2026-11-22에 직접 읽음)**
- depth 감독은 LiDAR가 아니라 3D 박스로 만든 전경 depth예요: "the depth D here is the foreground depth information provided using 3D boxes rather than the point cloud". LiDAR depth는 부록 C.2의 비교 실험에만 나와요. 그래서 BEVDepth + PD-BEV는 LiDAR depth 손실을 쓰지 않아요(공식 코드와 같아요).
- 식 3은 `V_ifv = sigmoid(F_height) · F_bev`예요. 그래서 PDBEV-12는 버그가 맞아요.
- 2D head 손실은 "Following CenterPoint, focal loss and L1 loss are used"라고만 적혀 있고, 정규화 방법은 없어요. CenterPoint 구현은 focal과 L1을 둘 다 positive 수로 나눠요. 그래서 PDBEV-13, PDBEV-14는 CenterPoint 방식을 의도로 봐요.
- 논문 식 8의 전체 손실은 `L = λs·L_det + λs·L_render + λs·L_pg + λs·L_ps + λt·L_con`이에요. source에서는 λs = 1이고, DG에서는 L_con을 쓰지 않아요. **car 전용 BEV 보조 손실(`bev_loss`)은 논문에 없어요.** 코드에만 있고, PDBEV-2와 PDBEV-17 때문에 공식 학습에서도 사실상 동작하지 않았어요.

**PCCR 적응 (버그는 아니지만 바꿔야 하는 것)**
- 카메라 이름 6개가 하드코딩돼 있어서 `data_config['cams']`를 무시해요. PCCR 리그는 모두 여기서 깨져요(`loading_UDA.py:492-501`, C11).
- 렌더링 상수가 하드코딩돼 있어요: 0.8 m 셀, depth bin 1..51(`view_transformer_pc.py:358-366`, C13).
- FOV 120° 리그(R5~R9)에서는 virtual depth 범위가 약 0.45~44.7 m라서 car 평가 범위 50 m보다 짧아요(M11).

**PD-BEV에서 해당 없는 것**
- `torch.max` depth 다운샘플(`view_transformer_pc.py:632-635`, B12 → C09)은 공식 `BEVDepth_DG` 기준선의 LiDAR depth 경로에만 있어요. PD-BEV 학습은 LiDAR depth를 쓰지 않아요(C10). 우리 vanilla BEVDepth는 원본 클래스를 복원해서 이 문제가 없어요.
- UDA 전용 경로: B10, B13~B17 (부호, 타깃 self-training, UDA 데이터셋).
- 인프라 문제(README 등): B00~B03, B18, B19, B24, B26.
- 이미 고친 것: B04 `/mnt/cfs` 덤프, B05 anomaly detection(커밋 `0be7094`), B06 `evaluate`의 `save_path`(커밋 `ed177d0`).

## PD-BEV 재현 체크 (`tools/bugrepro/pdbev_checks.py`)

CPU에서 합성 입력이나 실제 PCCR R1 mini 샘플로 돌려요. 각 체크는 버그가 깨뜨리는 동작을 직접 측정해요. 그래서 공식 코드에서는 FAIL, 고친 뒤에는 PASS가 나와요. PDBEV-11, PDBEV-15는 GPU 스모크 테스트에서 확인해요.

```bash
cd bevdepth && ~/miniconda3/envs/gen2e-bevdepth/bin/python ../tools/bugrepro/pdbev_checks.py
```

| ID | 체크 내용 | 고치기 전 (2026-11-22, PCCR 카메라 적응 `0318e03`만 적용) | 고친 뒤 | 커밋 |
|---|---|---|---|---|
| PDBEV-1 | 가로·세로가 다른 BEV에 셀마다 고유값을 넣고 IFV 렌더링으로 읽어요. | FAIL: `IndexError: index 3 is out of bounds for dimension 1 with size 3` (x와 y가 뒤바뀌어 y 크기를 넘음) | PASS: 30개 voxel 모두 올바른 셀 |  |
| PDBEV-3 | 위치 섭동을 (1.0, 0.5, 0.3) m로 고정하고, 렌더링 카메라 위치와 타깃 카메라 위치를 비교해요. | FAIL: 카메라 8대에서 최대 3.527 m 차이 | PASS: 카메라 8대에서 최대 0.000 m |  |
| PDBEV-4 | label −1 물체가 12개인 샘플에서 마지막 클래스(child) 채널의 peak 수를 세요. | FAIL: peak 17개, 기대 0개 | 이 커밋만으로는 peak 17 → 1개, 남은 1개는 범위·맵 밖 child 물체예요. PDBEV-5·6 수정 뒤 체크 결과는 그 행을 보세요. |  |
| PDBEV-5 | 같은 샘플의 2D heatmap peak 수를 기준 계산과 비교해요. 기준은 7클래스, BEV 범위 안, z > 1 m, 증강 후 맵 안이에요. | FAIL: peak 29개, 기대 8개 | PASS: peak 8개, 기대 8개 (같은 커밋에서 PDBEV-4 체크도 PASS: child 채널 peak 0개) |  |
| PDBEV-6/7 | 크롭으로 위쪽 물체가 맵 밖으로 나가게 한 뒤, mask 셀을 기준 3×3 창과 비교해요. | FAIL: 물체 창 밖 mask 셀 125개 | | |
| PDBEV-8 | z 8 m, height_num 3으로 view transformer를 만들고 z-bin이 덮는 범위를 재요. | FAIL: 3칸 × 2 m = 6 m (8 m 중) | PASS: 3칸 × 2.667 m = 8.0 m |  |
| PDBEV-9 | crop_w > 0인 상태로 flip한 샘플에서, 실제 intrinsic의 주점을 증강 행렬로 옮긴 원래 주점과 비교해요. 좌우 반전 자체는 post_rot이 담당해서 pinhole K로는 표현할 수 없어요. 그래서 주점 위치와 \ | fx'\ | , \ | fy'\ | 만 비교해요. | flip이면 실제 intrinsic의 주점을 `cx' = W_crop − cx'`로 옮겨요(`loading_UDA.py:595` 다음). 좌우 반전 자체는 공식대로 post_rot이 담당해요. 회전(±5.4°)도 공식대로 넣지 않아요. |  |
| PDBEV-10 | 앞 카메라에 pitch_aug, roll_aug를 각각 0.1 rad 주고, 카메라 좌표계에서의 회전축을 확인해요. | FAIL: pitch_aug는 카메라 z(광축), roll_aug는 카메라 x로 돌려요(반대여야 해요). | PASS: pitch_aug는 카메라 x축, roll_aug는 카메라 z축으로 돌려요. |  |
| PDBEV-12 | BEV_Aux 출력과 `sigmoid(F_height) · F_bev`를 비교해요. | FAIL: 최대 차이 0.261 (sigmoid 없음) | PASS: 최대 차이 0 (부동소수점 수준) |  |
| PDBEV-16 | PCCR 설정(crop_h (0, 0))으로 테스트 crop의 위쪽 행을 확인해요. 학습 crop은 아래 정렬이라 12여야 해요. | FAIL: 6 (가운데 정렬) | PASS: 위쪽 행 12 (학습과 같은 아래 정렬) |  |
| PDBEV-17 | conv 가중치를 무작위로 다시 뽑아 학습 뒤 상태를 흉내 내고, 세 보조 head의 sigmoid(출력) 최솟값을 재요. 마지막 연산이 ReLU면 어떤 가중치에서도 0.5 이상이에요. | FAIL: Img_Aux 0.500, Img_Aux_Dy 0.500, BEV_Aux height 0.500 (ReLU 바닥) | PASS: conv 가중치를 무작위로 다시 뽑았을 때 Img_Aux 0.029, Img_Aux_Dy 0.354, BEV_Aux height 0.083 (공식 코드는 같은 조건에서도 0.500) |  |

## UDGA: 재구현할 때 의도대로 만들 부분

UDGA는 코드 기반이 달라서 재구현해요([porting_plan.md](porting_plan.md) 3장). 위치는 `third_party/UDGA/` 기준이에요.

| ID | 위치 | 문제 | 발견 | 재구현에서 |
|---|---|---|---|---|
| UDGA-1 | `bevssl.py:62-74` | 재투영 점은 (D, w, h) 순서로 만들고 logit은 (D, h, w)로 펼쳐요. 그래서 점마다 다른 픽셀의 logit이 붙어요(실제 크기에서 점의 99.8%). | U02 (순수 Python 재현) → V06 confirmed (검증자도 직접 재현) | view transformer의 frustum `(D, fH, fW)` 점을 그대로 써서 순서를 맞춰요. |
| UDGA-2 | 타깃 `bevssl_unsupdepth.py:75` 등 vs lift `view_transformer.py:488-489` | depth 타깃은 d/k, lift는 bin/k예요. k ≈ 1일 때만 서로 맞아요. | U03 → V08 confirmed | 타깃과 lift에 같은 변환을 써요. |
| UDGA-3 | `bevssl.py:64, 126` | 재투영할 때 bin 번호를 미터 depth로 쓰고 pixel_const를 무시해요. | U04 | bin 중심의 실제 depth를 써요. |
| UDGA-4 | `bevssl.py:174, 212-218` | photometric 손실이 argmax와 정수 인덱싱을 거쳐서 그래디언트가 0이에요. 로그에만 찍혀요. | U05 → V07 confirmed | 논문 식 4처럼 그래디언트가 흐르게 구현해요. |
| UDGA-5 | `lyft_yawaug.py:62-83` | extaug가 박스와 외부 파라미터는 회전하지만 LiDAR 점은 회전하지 않아요. 증강된 절반은 depth GT가 틀려요. | U06 → V09 confirmed | 점도 같은 회전으로 옮겨요. |
| UDGA-6 | `bevssl_unsupdepth.py:135`, `bevssl.py:126` | mask가 비면 SmoothL1이 NaN이고, scatter 중복 쓰기는 GPU에서 비결정적이에요. | U07 → V11 partly_wrong (쌍 하나가 비는 건 괜찮고, 전체가 비면 NaN) | 빈 mask를 막고, 결정적으로 모아요. |
| UDGA-7 | `bevssl.py:33-45` | 카메라 쌍이 nuScenes 6대 링으로 하드코딩돼 있어요. | V05 | 캘리브레이션에서 겹침 비율로 쌍을 계산해요. 6대 링이면 원래 쌍 12개가 그대로 나와요. |
| UDGA-8 | `bevssl.py:55` vs `bevssl_unsupdepth.py:112` | 배치 크기를 두 곳에서 다르게 계산해요. 카메라 10대면 크래시가 나요. | W03 | 재구현에서는 한 곳에서만 계산해요. |
| UDGA-9 | `bevssl_unsupdepth.py:127-135` | 겹침 손실이 D축으로 sparse하고, 픽셀 mask를 89 bin 전체로 펼쳐요. | W01 | 구현할 때 정의를 정해서 기록해요. |

**UDGA에서 해당 없는 것:** LEDA/adapter 관련(U08, U09)은 target 라벨이 있는 적응 단계 전용이라 DG 표에서는 안 써요. CMKD(U10), 인프라(U00, U11~U16), NDS* 정규화(U14, 우리는 PCCR 평가를 써요)도 해당 없어요.
