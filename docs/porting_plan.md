# 포팅 계획

**대상 행:** BEVDepth(vanilla), BEVDepth + PD-BEV, BEVDepth + UDGA, BEVFormer-S(재현), BEVFormer-S + PD-BEV.

**출처:** 이 문서는 코드 분석 에이전트 5개와 교차 검증 에이전트 4개의 결과(2026-11-21~22)를 정리한 거예요.

- 주장마다 실제 코드로 다시 확인했어요. 틀린 주장은 고쳤어요.
- `파일:줄` 표기는 해당 디렉터리 기준이에요. `bevdepth/`는 PD-BEV 트리, `bevformer/`는 공식 BEVFormer, `third_party/UDGA/`는 UDGA 원본이에요.
- 효과가 큰 결정은 [decisions.md](decisions.md)의 "확인 필요" 표에 모아 뒀어요.

---

## 0. 공통 기반 작업

### 0-A. `bevdepth/`: PCCR 데이터셋 통합 (모든 BEVDepth 계열 행에 필요)

| 할 일 | 메모 |
|---|---|
| PCCR 어댑터 이식 | `third_party/pccr/models/BEVDet/pccr_dataset.py`의 `PCCRDataset`(NuScenesDataset 상속)를 `bevdepth/mmdet3d/datasets/`에 넣고 등록해요. PD-BEV 트리(dev2.0 @2ae6a02)의 `NuScenesDataset`에도 `ego_cam`과 `img_info_prototype` 인자가 있어서 API는 맞아요. 다만 PD-BEV가 `nuscenes_dataset.py`를 37줄 고쳤으니 호환성을 확인해야 해요. |
| `cams=None` 처리 | PCCR 공식 설정은 `data_config.cams=None`, `Ncams=None`이에요. 그런데 이를 처리하는 `PrepareImageInputs` 수정본은 **PCCR 레포에 없어요.** 업스트림 `choose_cams()`는 `len(None)`에서 실패해요. info에 있는 카메라 목록을 순서대로 쓰도록 구현하고, 이 부분이 † 수치와 차이가 날 수 있는 지점이라는 걸 기록해 둬요. |
| ego 기준 카메라 | 업스트림은 `CAM_FRONT`를 하드코딩해요. PCCR 어댑터는 `ego_cam='auto'`로 `CAM_BUMPER_FRONT` 등에서 골라요. |
| 학습·평가 스크립트 | PCCR `sequential_train.sh`/`sequential_test.sh`처럼, R1 학습 1회 + 14개 리그 test 평가로 바꿔요. 결과는 `tools/pccr_metric.py summarize`로 7-class mAP 표 행을 만들어요. |
| GPU 2장 | `tools/dist_train.sh <cfg> 2`. 기준 배치는 2×4예요(PCCR BEVDet). |

### 0-B. `bevdepth/`: vanilla BEVDepth를 표준대로 돌리기 위한 원본 클래스 복원

PD-BEV는 BEVDet 원본 16개 파일을 직접 고쳤어요(부록 A). vanilla BEVDepth에 영향을 주는 변경은 이렇게 확인됐어요.

- **`PointToMultiViewDepth`** (`mmdet3d/datasets/pipelines/loading.py`)
  - depth 라벨을 `depth * sqrt(100/fx² + 100/fy²) / c`(초점거리 450 px 기준 virtual depth)로 바꿔요.
  - 범위 필터는 `-100 < depth < 100`으로 바뀌었어요.
  - `img_inputs[4:8]`에 `act_intrins`(8번째 원소)를 요구해요.
- **`CenterHead`** (`mmdet3d/models/dense_heads/centerpoint_head.py`): 818줄이 바뀌었어요(`bev_loss`, `loss_pseudo`, `yaw_weight` 등).
- **PD-BEV의 `-pc-` BEVDepth 경로** (`BEVDepth_DG`)
  - DepthNet에 31차원 카메라 벡터를 쓰고, virtual-depth bin으로 lift해요. 표준 BEVDepth가 아니에요.
  - `torch.max` 버그 때문에 LiDAR depth 손실이 사실상 꺼져 있어요(검증 확인).

**계획:** BEVDet @2ae6a02의 원본 클래스를 **본문은 바이트 단위로 동일하게 두고 이름만 바꿔 별도 등록**해요(예: `PointToMultiViewDepthUpstream`). vanilla BEVDepth 설정은 이 클래스들만 쓰게 해요. 복원을 한 커밋으로 묶어서 공저자가 "원본과 동일"을 diff로 확인할 수 있게 해요.

- **단일 프레임 BEVDepth detector:** dev2.0에서 업스트림 `BEVDepth4D`를 인접 프레임 없이 쓸 수 있는지 먼저 확인해요. 안 되면 업스트림 `BEVDet`에 DepthNet 손실만 붙인 얇은 클래스를 만들어요.
- **PD-BEV의 `BEVDepth_DG`는 vanilla로 쓰면 안 돼요.** PD-BEV의 8원소 입력 형식에서만 동작해요.

### 0-C. `bevformer/`: PCCR 통합 (BEVFormer-S 계열 행에 필요)

| 할 일 | 메모 |
|---|---|
| info 변환기 | PCCR info를 BEVFormer 형식으로 바꿔요. 필요한 키는 `sensor2lidar_*`, `cam_intrinsic`, `can_bus`(0 벡터 18개 + ego pose), `gt_velocity`, `valid_flag`예요. PCCR info에는 `next`, `frame_idx`, `can_bus`가 없어서 `CustomNuScenesDataset.get_data_info`가 KeyError를 내요(`nuscenes_dataset.py:114-117`). |
| 박스 좌표계 | PCCR은 mmdet3d 1.0 형식 `[x,y,z,l,w,h,yaw]`이고, BEVFormer(mmdet3d 0.17.1)는 SECOND 형식 `[x,y,z,w,l,h,-yaw-π/2]`예요. 변환과 단위 테스트(GT 투영 오버레이)가 필요해요. |
| 카메라 수 | `num_cams=6`이 하드코딩돼 있어요(`transformer.py:171`, `spatial_cross_attention.py:152/157`). **카메라 8대인 R1 학습부터 크래시가 나요.** 평가 리그는 카메라가 4~10대예요. SCA가 `key.shape[0]`에서 카메라 수를 읽게 고쳐요. SCA 가중치는 카메라별이 아니라서 안전해요. |
| `cams_embeds` | 학습 가능한 `[num_cams, C]` 임베딩이라 카메라 수가 바뀌면 깨져요. `use_cams_embeds=False`로 하려면 DDP에서 `find_unused_parameters=True`가 필요하거나, 쓸 때만 파라미터를 만들어야 해요(`transformerV2.py:78-79` 방식). **기존 BEVFormer-S 행과 같은 방식이어야 해요(확인 필요).** |
| 단일 프레임 | `queue_length=1`이면 크래시가 나요. `prev_bev`를 `len_queue > 1`일 때만 계산하도록 고쳐요(`bevformer.py:217-222`). `video_test_mode=False`로 둬요. |
| 클래스 | 학습 클래스는 7개예요(traffic 2종 제외, 2026-11-22 결정). head와 `bbox_coder`의 `num_classes`를 같이 바꿔야 해요(NMSFreeCoder는 `idx % num_classes`로 디코딩해요). |
| 박스 z | `gt_boxes`의 z는 박스 중심이에요. mmdet3d 0.17.1 기본 `get_ann_info`(`origin=(0.5, 0.5, 0.5)`)를 그대로 써요. PCCR의 PETR·BEVFusion 로더는 `origin=(0.5, 0.5, 0)`이라 박스가 h/2 떠요. **그 로더를 템플릿으로 쓰면 안 돼요**([pccr_dataset.md](pccr_dataset.md) "박스 z 확인"). |
| 좌표 기준과 차량 높이 | PCCR LIDAR_TOP은 ego와 회전이 같고 높이만 차량마다 달라요. R1~R3·R7은 1.86 m, R4·R5·R8은 1.50 m, R6은 4.80 m, R9는 4.18 m(x=+2.6)예요. BEVFormer는 LiDAR 좌표계를 쓰기 때문에, 리그가 바뀌면 지면 높이가 최대 3.3 m 어긋나요. bevformer_small의 pillar 샘플 높이도 지면 기준으로 리그마다 달라져요(2026-11-22 확인). LiDAR 좌표계를 그대로 쓸지, 지면 기준 고정 좌표계로 바꿀지 정해야 해요. **기존 BEVFormer-S 행 방식과 맞춰야 해요(확인 필요).** |
| 해상도 | 모든 PCCR 카메라는 1280×720이에요. 공식 small은 0.8 스케일(1024×576)을 써요. |
| 배치 | MSDA의 `im2col_step=64` 때문에 `bs × num_cams ≤ 64`(또는 64의 배수)여야 해요. GPU당 1샘플로 카메라 10대까지는 괜찮아요. 전체 배치 8이 필요하면 `GradientCumulativeOptimizerHook`을 써요. |
| 기타 버그 | TSA `value[:bs]`, SCA batch-0 mask 버그는 `samples_per_gpu > 1`일 때만 문제가 돼요. |

---

## 1. BEVDepth (vanilla) — 규모 M

1. 0-A, 0-B를 끝내요.
2. `bevdepth/configs/gen2e/bevdepth-r50-pccr.py`를 만들어요. PCCR BEVDet 설정(입력 384×704, grid 0.8 m, AdamW 2e-4, 24 epoch, 2×4 배치, EMA)에 BEVDepth 요소를 더해요: 원본 `LSSViewTransformerBEVDepth`, DepthNet, LiDAR 기반 원본 `PointToMultiViewDepth`, depth loss weight. head는 7-class 단일 task예요.
3. R1에서 학습하고 14개 리그 test로 평가한 뒤 7-class mAP를 집계해요.
4. **확인용 기준:** 같은 설정에서 depth를 끄면 PCCR의 BEVDet† 행(R1 25.2)과 비슷해야 해요. 파이프라인 검증용으로 선택 사항이에요.

## 2. BEVDepth + PD-BEV — 규모 M

**PD-BEV가 하는 일** (분석과 검증 결과)

- **테스트:** PD-BEV 체크포인트는 `BEVDepth_DG` 구성으로 평가해요(`configs/bevdet_our/bevdepth-r50-cbgs-pc-*.py`). `PCBEV_DG.simple_test`는 고장 나 있어요(`pcbev.py:62-69, 110-124`).
- **학습 때만 추가되는 것:**
  1. 이미지 feature 위 2D CenterNet head (L_ps)
  2. BEV → implicit foreground volume (IFV)
  3. 실제 카메라와 섭동된 가상 카메라에서의 ray rendering (MVSR)
  4. 렌더링된 PV feature 위 2D head (L_render)
  5. car-only BEV heatmap 보조 손실
  6. 박스 중심 depth 지도학습 (L_pg). PD-BEV 손실은 LiDAR depth를 쓰지 않아요.
- **"pc"의 의미:** 설정 이름의 "pc"는 point cloud가 아니라 "PCBEV(`view_transformer_pc.py`) 기반"이라는 뜻이에요.

**할 일**

1. PCCR용 DG 설정을 만들어요: `configs/gen2e/pdbev-r50-pccr-dg.py`(학습, `PCBEV_DG`)와 평가용 `BEVDepth_DG` 설정.
2. 카메라 이름과 내부 파라미터가 하드코딩된 곳을 일반화해요(`loading_UDA.py:492-501` 등). DG 경로에서 `Cam_Num=6`이 실제로 걸리는 곳은 없어요. `pcbev.py:374`는 UDA 경로에만 있어요.
3. 환경 정리
   - `tools/test.py:245`의 `/mnt/cfs` 덤프를 없애요. root가 아니면 크래시가 나요.
   - `tools/train.py:120`의 `set_detect_anomaly(True)`를 없애요. 느려지기만 해요.
   - `timm(≤0.6)`과 `einops`는 env에 설치했어요.

**결정이 필요한 것**

- **head 구성:** `bev_loss`는 head task 0이 클래스 1개일 때만 동작해요(`centerpoint_head.py:656`의 `squeeze(1)`). PCCR 기준 head는 9-class 단일 task라 형태가 맞지 않아 크래시가 나요. car 단독 task를 앞에 두는 다중 task head로 바꿀지, `bev_loss`를 일반화할지 정해야 해요.
- **원본 재현 vs 버그 수정:** 아래 버그를 그대로 둘지(공식 코드 재현) 고칠지(논문 의도) 정해야 해요.
  - 렌더러의 BEV 샘플링 축이 뒤바뀌어 있어요([Z,X,Y]로 인덱싱).
  - BEV 보조 손실의 positive mask가 `eq(0)`이에요.
  - 가상 카메라 렌더링과 타깃이 다른 translation을 써요. 오차는 무작위이고 수 m까지 생겨요.
  - label -1(무시 클래스)이 마지막 클래스 채널에 그려져요(`loading_UDA.py:929, 1063`). PCCR 어댑터는 클래스 밖 이름에 -1을 내보내요.
  - 2D 타깃이 범위·이름 필터 전의 `ann_infos`로 만들어져요(`loading_UDA.py:860, 998`).
  - 음수 인덱스 wrap-around로 타깃이 엉뚱한 위치에 써져요(`loading_UDA.py:938-952`).
  - IFV z-bin이 `int(z_interval/height_num)`라서 `height_num=3`이면 8 m 중 6 m만 덮어요(`view_transformer_pc.py:253`).
- **PCCR 리그와의 범위 차이**
  - PD-BEV의 yaw 섭동(±23°)은 R1-r(±105°)보다 훨씬 작아요. xyz 섭동(±2 m)은 R1-t(≤0.6 m)보다 커요.
  - FOV 120° 리그(R5~R9)에서는 virtual-depth 범위가 약 0.45~44.7 m라서 car 평가 범위(50 m)보다 짧아요.
  - 원 논문 값을 유지할지 정해야 해요.

## 3. BEVDepth + UDGA — 규모 L (재구현)

UDGA 원본은 **BEVDet master @b17b6e8(mmdet3d 0.17.2)용 오버레이**라서, mmdet3d 1.0.0rc4인 PD-BEV 트리에 파일을 그대로 복사할 수 없어요. 아래 구성요소를 새로 구현해요. 원본은 `third_party/UDGA/`에 참고용으로 둬요.

| 구성요소 | 계획 |
|---|---|
| Multi-view Overlap Depth Constraint | view transformer의 frustum `(D,fH,fW)`에서 점을 만들어요. 원본의 축 뒤바뀜 버그(`bevssl.py:62-74`)도 이걸로 해결돼요. 카메라 i에서 역투영해서 카메라 j로 투영하고, 겹치는 픽셀에 SmoothL1을 적용해요. |
| 카메라 쌍 | 원본은 nuScenes식 6대 링을 하드코딩했어요. 4~10대 리그에 맞게 **캘리브레이션에서 겹침 비율로 쌍을 계산**해요. 6대 링이면 원본의 directed pair 12개가 그대로 나와요. 겹치는 곳이 전혀 없을 때를 대비한 가드도 둬요. |
| Photometric loss | 공개 코드의 이 항은 **그래디언트가 0**이라서, 빼도 공개 코드와 동작이 같아요. 논문 식 4로 구현하면 원본과 달라지는 부분으로 표시해요. |
| pixel_const | 공개 코드는 타깃을 d/k로, lift는 bin/k로 해서 서로 맞지 않아요. 선택 사항이에요. |
| extaug | BDA yaw를 {0, ±90°, 180°}에서 고르는 방식으로 구현해요. 원본에는 LiDAR 점을 회전 좌표계로 옮기지 않는 버그가 있어요. |
| LEDA (adapter) | target 라벨이 있을 때만 쓰는 구성요소라서 DG 표에서는 쓰지 않아요. 원본에는 adapter 파라미터가 등록되지 않는 버그와 `requries_grad` 오타가 있어요. |
| 좌표계 0.17 → 1.0 | 박스 `(x,y,z,w,l,h,-yaw-π/2)` → `(x,y,z,l,w,h,yaw)`. yaw를 고치는 증강은 부호가 바뀌어요. |
| depth 지도학습 방식 | dev2.x(softmax BCE, GT min-pool, weight 약 3)와 master(sigmoid BCE, weight 100) 중에서 골라요. |

**결정이 필요한 것**

- **재현 범위:** 축 뒤바뀜(점의 99.8%가 다른 픽셀의 값을 받음) 같은 동작을 바꾸는 버그를 재현할지 고칠지 정해야 해요.
- **손실 가중치:** 논문에 λ 값이 없어요.
- **epoch:** 논문은 24, 코드는 32예요.

참고로 카메라 10대 리그에서는 원본 코드의 배치 계산 두 개가 서로 달라서 크래시가 나요(`bevssl.py:55` vs `bevssl_unsupdepth.py:112`). 재구현에서는 이 문제가 없어요.

## 4. BEVFormer-S (vanilla 재현) — 규모 M

1. 0-C를 끝내요.
2. `bevformer_small` 구조에 단일 프레임 설정으로 `bevformer_small_static_pccr.py`를 만들어요.
   - 원본 설정: R101-DCN, BEV 150×150, encoder 3층, FPN 1레벨, load_from은 `r101_dcn_fcos3d_pretrain.pth`예요.
3. R1에서 학습해서 **표의 BEVFormer-S 행(R1 21.1, R1-c6 19.6, R1-c10 26.2 …)이 재현되는지 먼저 확인해요.** 재현되면 그 설정 위에 PD-BEV를 붙여요.

## 5. BEVFormer-S + PD-BEV — 규모 L (재구현)

PD-BEV 공식 코드는 BEVDepth 전용이라(README.md:50) 재구현이 필요해요.

1. `BEVFormerPDBEV(BEVFormer)`를 만들어요. 구성은 BEV → voxel lift(높이 4 bin), `lidar2img` 역행렬 기반 frustum 렌더러, bin 합산, 렌더링 PV feature 위 2D head, `img_feats` 위 2D head, BEV heatmap head예요.
2. `forward_pts_train`에 보조 손실을 더해요. 추론은 바뀌지 않아요.
3. 파이프라인에 카메라 섭동(`lidar2img_aug`)과 2D 타깃 생성을 추가해요.
4. 상수를 옮겨요. 0.8 m 셀은 102.4/150 m로, LSS frustum은 `lidar2img` 기반으로 바꿔요. BEV 축은 둘 다 `[y, x]`예요. 차이는 기준 좌표계(LiDAR vs ego)와 셀 크기예요.
5. virtual depth처럼 BEVDepth 전용인 부분은 빼고, 빠진 부분을 문서에 남겨요.
6. DDP 주의: 보조 파라미터는 모두 그래디언트를 받아야 해요(`find_unused_parameters=False` 기준).

---

## 6. 결정이 필요한 사항 요약

| # | 항목 | 선택지 |
|---|---|---|
| 1 | 보고 지표 | **결정(2026-11-22):** 7-class로 학습하고 7-class mAP로 보고 |
| 2 | 기존 BEVFormer-S PCCR 코드 | 있으면 재사용 / 없으면 새로 만들고 재현부터 확인 |
| 3 | BEVFormer `cams_embeds`, 좌표 기준(LiDAR vs 지면 기준 고정 좌표계) | 기존 BEVFormer-S 행 방식과 동일하게 |
| 4 | 학습 하이퍼파라미터 | **결정(2026-11-22):** BEVDepth 계열은 PCCR BEVDet 기준. BEVFormer-S는 2번과 함께 정해요 |
| 5 | PD-BEV head 구성 (`bev_loss` 제약) | car 단독 task를 앞에 두는 다중 task / `bev_loss` 일반화 |
| 6 | PD-BEV·UDGA 공식 코드 버그 | 공식 코드 재현(버그 유지) / 논문 의도대로 수정 (어느 쪽이든 문서화) |
| 7 | PD-BEV 섭동 범위, UDGA λ·epoch | 원 논문 값 / PCCR에 맞춰 조정 |

---

## 부록 A. PD-BEV가 원본 BEVDet(dev2.0 @2ae6a02)에서 바꾼 파일

비교 방법은 `git diff`(BEVDet @2ae6a02 ↔ `bevdepth/`)이고, `__pycache__`, `.so`, `.mim`은 제외했어요.

**수정 (`mmdet3d/` 16개)**

| 파일 | 바뀐 줄 수 |
|---|---|
| `apis/train.py` | — |
| `core/evaluation/lyft_eval.py` | — |
| `datasets/__init__.py` | — |
| `datasets/custom_3d.py` | — |
| `datasets/dataset_wrappers.py` | — |
| `datasets/lyft_dataset.py` | — |
| `datasets/nuscenes_dataset.py` | 37 |
| `datasets/pipelines/__init__.py` | — |
| `datasets/pipelines/loading.py` | 435 |
| `models/backbones/__init__.py` | — |
| `models/dense_heads/centerpoint_head.py` | 818 |
| `models/detectors/__init__.py` | — |
| `models/detectors/bevdet.py` | 7 |
| `models/necks/__init__.py` | — |
| `models/necks/view_transformer.py` | 21 |
| `version.py` | — |

`tools/{train,test}.py`와 `tools/dist_{train,test}.sh`도 수정됐어요.

**추가 (`mmdet3d/` 18개)**

- 데이터셋: `CARLA_dataset.py`, `DeepAccident_dataset.py`, `lyft_dataset_DA.py`, `lyft_dataset_pkl.py`, `nuscenes_dataset_DA.py`, `shift_dataset.py`, `UDA_DA_LYFT_dataset.py`, `UDA_DA_Nus_dataset.py`
- 파이프라인: `loading_plug.py`, `loading_UDA.py`
- backbone: `context_cluster.py`, `context_cluster2.py`, `context_cluster3.py` (미사용이지만 import 시점에 `timm`·`einops`가 필요해요)
- detector: `bevdepth.py` (BEVDepth_DG), `bevdet_intri.py`, `pcbev.py` (PCBEV_DG / PCBEV_UDA)
- neck: `img_aux.py`, `view_transformer_pc.py`

**삭제됨 (원본에는 있음):** `setup.py`, `setup.cfg`, `LICENSE`, `MANIFEST.in`, `requirements.txt`. 앞의 다섯 개는 커밋 `5efe61f`에서 원본 그대로 복원했어요.

## 부록 B. 분석 원자료

에이전트 원본 결과(JSON)는 세션 작업 폴더에만 있고 레포에는 넣지 않았어요. 필요하면 요약을 이 문서에 계속 반영해요.
