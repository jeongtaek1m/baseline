# PCCR 데이터셋

**Plentiful CARLA Camera Rigs (PCCR):** Bader et al., *Understanding Cross-Rig Generalization in Automotive Perception*, ECCV 2026 (arXiv 2606.27554).

| 항목 | 값 |
|---|---|
| HF 데이터셋 | [timb2001/plentiful-carla-camera-rigs](https://huggingface.co/datasets/timb2001/plentiful-carla-camera-rigs), 리비전 `18edf4f4bbc94840d7e4026f6f47b063ba173f04`로 고정 |
| 라이선스 | CC-BY-NC-4.0 (비상업) |
| 벤치마크 코드 | [BaderTim/plentiful-carla-camera-rigs](https://github.com/BaderTim/plentiful-carla-camera-rigs) @ `6798206`, `third_party/pccr/`에 원본 그대로 둠 |
| 서버 위치 | `/data2/PCCR/<RIG>/`, 레포에서는 `data/pccr`(심볼릭 링크)로 접근 |
| 용량 | zip 160.8 GB. 압축을 풀면 리그당 7.4~18 GB, 전체 193 GB (2026-11-22 확인) |

## 받기

```bash
bash tools/download_pccr.sh /data2/PCCR          # 14개 리그 전부
bash tools/download_pccr.sh /data2/PCCR R1 R4    # 일부만
```

- **검증:** 리그마다 크기와 sha256을 확인해요(`tools/pccr_sha256.txt`). 끝나면 `<RIG>/.pccr_done`을 남겨요.
- **다시 실행:** 끝난 리그는 건너뛰고, 받다 만 zip은 이어 받아요.
- **로그:** `/data2/PCCR/download.log`에 남아요.

## 리그 구조

```
/data2/PCCR/<RIG>/
├── <RIG>_infos_{train,val,test,mini}.pkl    mmdet3d 형식 info (박스는 mmdet3d 1.0 좌표계)
├── samples/CAM_*/  samples/LIDAR_TOP/      이미지 1280×720, LiDAR(32채널)
├── v1.0-{trainval,test,mini}/              nuScenes 형식 JSON 테이블
├── can_bus/                                nuScenes 형식 CAN bus JSON
└── logs/
```

## 14개 리그 정의

`python3 tools/pccr_rig_summary.py`의 출력이에요. 원자료는 `third_party/pccr/pccr/configs/rigs/<RIG>.json`(CARLA 좌표: x 앞, y 오른쪽, z 위, 단위 m)이에요.

| Rig | Vehicle | #Cams | FOV (deg) → f @1280px | Cam height (m) | Cam pitch (deg) | LiDAR xyz (m) |
|---|---|---|---|---|---|---|
| R1 | audi.etron | 8 | 65 → 1005 | 0.64-1.40 | 0 | -0.46, 0.00, 1.86 |
| R1-c6 | audi.etron | 6 | 65 → 1005 | 0.64-1.40 | 0 | -0.46, 0.00, 1.86 |
| R1-c10 | audi.etron | 10 | 65 → 1005 | 0.64-1.40 | 0 | -0.46, 0.00, 1.86 |
| R1-f | audi.etron | 8 | 90 → 640 | 0.64-1.40 | 0 | -0.46, 0.00, 1.86 |
| R1-r | audi.etron | 8 | 65 → 1005 | 0.64-1.40 | 0 | -0.46, 0.00, 1.86 |
| R1-t | audi.etron | 8 | 65 → 1005 | 0.90-1.46 | 0 | -0.46, 0.00, 1.86 |
| R2 | audi.etron | 6 | 80 → 763, 110 → 448 | 0.50-1.36 | 0 | -0.46, 0.00, 1.86 |
| R3 | audi.etron | 10 | 75 → 834 | 0.50-1.48 | 0 | -0.46, 0.00, 1.86 |
| R4 | audi.tt | 8 | 65 → 1005 | 0.68-1.04 | 0 | -0.22, 0.00, 1.50 |
| R5 | audi.tt | 4 | 120 → 370 | 0.72-1.34 | 0, 10, 18 | -0.22, 0.00, 1.50 |
| R6 | carlamotors.firetruck | 9 | 75 → 834, 120 → 370 | 0.86-3.34 | 0 | -0.36, 0.00, 4.80 |
| R7 | audi.etron | 7 | 70 → 914, 120 → 370 | 0.50-1.42 | 0 | -0.46, 0.00, 1.86 |
| R8 | audi.tt | 6 | 65 → 1005, 100 → 537, 120 → 370 | 0.60-1.38 | 0, 20 | -0.22, 0.00, 1.50 |
| R9 | carlamotors.european_hgv | 7 | 75 → 834, 120 → 370 | 1.20-3.34 | 0 | 2.60, 0.00, 4.18 |

### 통제 변형 (R1 기준)

| Rig | R1과의 차이 |
|---|---|
| R1-c6 | FENDER_BACK_LEFT·RIGHT를 뺐어요 (카메라 6대). |
| R1-c10 | MIRROR_FRONT_LEFT·RIGHT를 추가했어요 (카메라 10대). |
| R1-f | 카메라 8대의 FOV를 모두 65°에서 90°로 바꿨어요 (f 1005 → 640 px). **f = FOV.** |
| R1-r | 펜더 카메라 4대를 yaw ±104~107.5° 돌렸어요. 앞 펜더는 뒤쪽을, 뒤 펜더는 앞쪽을 봐요. **r = rotation.** |
| R1-t | 카메라 위치를 옮겼어요(z 최대 +0.60 m, xy ±0.5 m). **t = translation.** |

> HF README에는 "R1: 6 cameras", "R1-f/r/t: front-facing / rear-facing / top-mounted"라고 적혀 있지만, **실제 데이터와 리그 정의 JSON은 위 표와 같아요.** R1은 카메라 8대이고, f/r/t는 각각 FOV, 회전, 위치 변경이에요.

## 클래스

9개 클래스: car, truck, bus, bicycle, motorcycle, adult, child, traffic_sign, traffic_light.
표 지표는 이 중 traffic 2종을 뺀 **7-class mAP**예요([evaluation_protocol.md](evaluation_protocol.md)). 우리 포팅은 학습도 이 7개 클래스로 해요([decisions.md](decisions.md), 2026-11-22).

## 박스 z 확인 (2026-11-22)

`tools/pccr_box_z_check.py`로 확인했어요. 이 도구는 세 가지를 해요.

- `stats`: 14개 리그 test split의 박스 전부를 봐요.
- `lidar`: R1·R4·R6·R9에서 test 150프레임씩, 25 m 이내 정지 물체를 LiDAR 점과 비교해요.
- `render`: 박스를 카메라 이미지에 투영해요.

**결론:** 박스 z는 정확해요. 리그마다 달라지는 건 LiDAR 좌표계 원점의 높이뿐이에요.

| 항목 | 결과 |
|---|---|
| z의 의미 | `gt_boxes`의 z는 박스 **중심**이에요(nuScenes `Box.center`, `nuscenes_converter.py:378`). mmdet3d 규약대로 `origin=(0.5, 0.5, 0.5)`로 읽어야 해요. |
| 좌표계 | `gt_boxes`는 LiDAR 좌표계, BEVDet용 `ann_infos`는 ego 좌표계예요. 둘은 lidar2ego 이동만큼만 달라요(최대 차이 7e-15). lidar2ego 회전은 14개 리그 모두 identity예요. |
| ego 원점 | 4개 차종 모두 지면에 있어요. 차량류 박스 바닥의 ego z 중앙값이 −0.016~+0.002 m예요. |
| 지면 접촉 (LiDAR) | 주변 지면 대비 박스 바닥의 중앙값이 차량류 −0.025~+0.002 m, 보행자 +0.023~+0.036 m예요. 주변 지면은 박스 둘레 0.3~1.5 m 띠에 있는 점 z의 10% 분위수로 잡았어요. |
| 박스 높이 (LiDAR) | 위에서 내려다보는 높은 LiDAR(R6·R9)에서는 박스 윗면과 지붕 점의 차이가 1~3 cm예요(car +0.025 m). 낮은 LiDAR(R1·R4)는 빔이 성겨서 +0.09~0.10 m예요. |
| 보행자 +0.18 m | 보행자 박스 바닥의 ego z 중앙값은 +0.18 m예요. 연석 위 인도에 서 있기 때문이고, 주변 지면 대비로는 +3 cm예요. |
| 이미지 투영 | R1과 R6 모두 박스가 물체에 맞아요. |

이미지는 git에서 빠진 `logs/zcheck/`에 있어요. 다시 만들려면 이렇게 해요.

```bash
python -I tools/pccr_box_z_check.py render R1 logs/zcheck/box_z_R1.jpg
```

**LiDAR 좌표계의 z는 리그마다 달라요.** 같은 물체도 LiDAR 높이만큼 z가 바뀌어요.

| 차종 (리그) | LiDAR 높이 | 지면의 LiDAR z | car 중심 z 중앙값 (LiDAR) |
|---|---|---|---|
| audi.etron (R1, R1-*, R2, R3, R7) | 1.86 m | −1.86 | −1.07 |
| audi.tt (R4, R5, R8) | 1.50 m | −1.50 | −0.71 |
| european_hgv (R9) | 4.18 m | −4.18 | −3.39 |
| firetruck (R6) | 4.80 m | −4.80 | −4.01 |

### 모델별 영향

**BEVDepth 계열 (`bevdepth/`):** 고칠 게 없어요.

- `ann_infos`(ego 좌표계, 지면 z=0)를 `origin=(0.5, 0.5, 0.5)`로 읽어요(`loading.py:1490`). 그래서 리그가 바뀌어도 z 기준이 같아요.
- PCCR BEVDet의 grid z가 [−6, 6] m라 버스 윗면까지 들어가요.

**BEVFormer 계열 (`bevformer/`):** z 해석은 맞지만 좌표계를 정해야 해요.

- mmdet3d 0.17.1도 `origin=(0.5, 0.5, 0.5)`로 읽어서 z 해석 자체는 맞아요.
- 문제는 LiDAR 좌표계로 학습한다는 점이에요. 리그마다 지면 높이가 달라요.
- bevformer_small은 pc_range z [−5, 3]에서 높이 4점(LiDAR z −4.5, −2.17, +0.17, +2.5)을 샘플해요. 이 높이를 지면 기준으로 바꾸면 아래 표와 같아요.

| 리그 | 지면 기준 샘플 높이 (m) |
|---|---|
| R1 계열 (LiDAR 1.86 m) | −2.64, −0.31, +2.03, +4.36 |
| R4, R5, R8 (1.50 m) | −3.00, −0.67, +1.67, +4.00 |
| R9 (4.18 m) | −0.32, +2.01, +4.35, +6.68 |
| R6 (4.80 m) | +0.30, +2.63, +4.97, +7.30 |

- 그래서 R1에서 학습한 모델이 R6·R9에서는 다른 높이의 특징을 봐요. 카메라 리그와 상관없는 좌표계 차이가 결과에 섞일 수 있어요.
- **결정 (2026-11-22):** 차량 고정 좌표계로 학습해요. 원점은 모든 차량에서 R1의 LiDAR 장착 위치(ego 기준 x −0.46, z +1.86 m)예요.
  - 그래서 모든 리그에서 샘플 높이가 위 표의 R1 계열 줄과 같아요.
  - etron 리그는 원래 LiDAR 좌표계와 수치까지 같아요.
  - 근거는 [decisions.md](decisions.md)에 있어요.

**PCCR 공식 PETR·BEVFusion 로더:** 박스가 h/2만큼 떠서 학습돼요. **BEVFormer 쪽 템플릿으로 가져오면 안 돼요.**

- 두 로더는 `gt_boxes`를 `origin=(0.5, 0.5, 0)`, 즉 z를 바닥으로 읽어요(`models/PETR/pccr_dataset.py:404`, `models/BEVFusion/pccr_dataset.py:380`).
- 그래서 학습 박스가 h/2만큼 위로 떠요(car 0.77 m, adult 0.93 m, bus 1.28 m).
- 그래도 † 수치는 영향을 받지 않아요. nuScenes 지표가 z를 보지 않기 때문이에요(BEV 2D 중심 거리, ASE는 중심을 맞춘 뒤 IoU).

## PCCR 공식 모델의 좌표계 (2026-11-22)

PCCR은 모델마다 **원래 레포의 기본 좌표계를 그대로** 썼고, 리그별 높이 보정은 하지 않았어요. 어댑터·설정은 `third_party/pccr/models/`에서, 모델 내부 동작은 업스트림 코드에서 확인했어요(아래 "근거").

| 모델 | 박스·BEV 좌표계 | 높이 방향 샘플링 | LiDAR 높이가 바뀔 때 | 박스 z 읽기 | 공식 7-class mAP R1 / R3 / R7 / R6 / R9 |
|---|---|---|---|---|---|
| BEVDet† | ego (모든 리그에서 지면 z=0) | z 1칸, −6~6 m | 영향 없음 | 중심 (맞음) | 25.2 / 9.0 / 11.2 / 8.2 / 5.1 |
| BEVFusion-C† | LiDAR | z 1칸, LiDAR 기준 −10~10 m | 거의 없음 (칸 하나가 높이를 다 덮어요) | 바닥 (h/2 뜸) | 26.0 / 14.6 / 12.8 / 10.6 / 6.5 |
| PETR† | LiDAR | 3D 위치 임베딩(z를 −10~10 m로 정규화) + 학습된 3D 기준점 900개(z −6~6 m) | 있음: 위치 임베딩과 기준점의 지면 대비 높이가 R6에서 2.9 m 어긋나요 | 바닥 (h/2 뜸) | 21.1 / 10.7 / 5.8 / 0.0 / 0.0 |
| Fast-BEV† | LiDAR | voxel 6층, LiDAR z −3.99 / −2.66 / −1.33 / 0 / +1.33 / +2.66 m | 큼: 층마다 채널이 따로라서 층의 높이 의미가 바뀌어요 | 중심 (맞음) | 12.9 / 9.0 / 9.0 / 0.0 / 0.0 |

Fast-BEV voxel 층의 지면 기준 높이예요.

| 리그 | 지면 기준 층 높이 (m) |
|---|---|
| R1 계열 (LiDAR 1.86 m) | −2.13, −0.80, +0.53, +1.86, +3.19, +4.52 |
| R4, R5, R8 (1.50 m) | −2.49, −1.16, +0.17, +1.50, +2.83, +4.16 |
| R9 (4.18 m) | +0.19, +1.52, +2.85, +4.18, +5.51, +6.84 |
| R6 (4.80 m) | +0.81, +2.14, +3.47, +4.80, +6.13, +7.46 |

**보이는 패턴**

- LiDAR 좌표계에서 높이 여러 층을 샘플링하는 PETR과 Fast-BEV만, LiDAR가 높은 R6·R9에서 0.0이에요. 높이가 1칸인 BEVDet(ego)과 BEVFusion(LiDAR)은 같은 리그에서 5~11점이에요.
- LiDAR 높이가 R1과 같은 R3·R7에서는 PETR·Fast-BEV도 R1 성능의 27~70%를 유지해요. 그래서 카메라 변화보다 높이 차이 때문일 가능성이 높아요. 다만 인과를 직접 실험하지는 않았어요.
- BEVFormer-S도 LiDAR 좌표계에서 pillar마다 높이 4점을 샘플링해요(위 표). 그래서 LiDAR 좌표계로 학습하면 R6·R9에서 PETR·Fast-BEV처럼 떨어질 가능성이 높아요.

**박스 형식 문제 (PETR·BEVFusion·Fast-BEV, NDS만 영향)**

- 세 어댑터는 mmdet3d 1.x 형식 `gt_boxes`(l, w, h, yaw)를 변환 없이 학습 타깃으로 써요.
- 평가할 때는 예측을 0.x 식(`yaw = -yaw - π/2`, `size = dims`)으로 되돌려요(`PETR/pccr_dataset.py:436-438`, `BEVFusion/pccr_dataset.py:412-414`, `Fast-BEV/pccr_dataset.py:443-444`).
- 그래서 평가 박스의 w/l이 뒤바뀌고 yaw가 반전돼요. 공식 결과에서도 확인돼요(R1).
  - car 크기 오차(ASE)가 0.73~0.74예요. car 크기에서 w/l을 바꿨을 때의 이론값이 0.73이에요.
  - 방향 오차(AOE)는 모든 클래스에서 1.35~1.70으로 무작위 수준(≈π/2)이에요.
  - 같은 지표가 BEVDet은 ASE 0.22 / car AOE 1.00, 우리 BEVDepth는 ASE 0.22 / car AOE 0.69예요.
- mAP는 중심 거리만 보기 때문에 **표의 † 수치에는 영향이 없어요.** NDS·mASE·mAOE만 틀려요.

**근거**

- **BEVDet**
  - `ann_infos`(ego)를 `LoadAnnotations`·`BEVAug`가 `origin=(0.5, 0.5, 0.5)`로 읽어요(업스트림 dev3.0 `loading.py`).
  - grid는 `z=[-6, 6, 12]`예요(`bevdet-r50-pccr.py:486-490`).
  - 평가는 ego 박스를 `ego_cam`의 ego pose로 global로 옮겨요(`BEVDet/pccr_dataset.py:250-264`). w/l 순서도 `box[[4, 3, 5]]`로 맞춰요.
- **PETR**
  - `lidar2img`를 `sensor2lidar`로 만들어요(`PETR/pccr_dataset.py:349-363`).
  - `position_range=[-61.2, -61.2, -10, 61.2, 61.2, 10]`, `pc_range` z는 [−6, 6]이에요(설정 345·377·407행).
  - 위치 임베딩은 `img2lidar`로 만든 3D 좌표를 `position_range`로 정규화해요(업스트림 `petr_head.py`).
- **BEVFusion**
  - `zbound=[-10, 10, 20]`이라 z가 1칸이에요(`configs.yaml:532-535`).
  - geometry는 `camera2lidar`로 계산해요(업스트림 `vtransforms/base.py`).
- **Fast-BEV**
  - `n_voxels=[250, 250, 6]`, `voxel_size=[0.64, 0.64, 1.33]`예요(설정 84-85행).
  - `origin`은 어댑터에서 0이고(`Fast-BEV/pccr_dataset.py:282`), `KittiSetOrigin`이 pc range 중심(0, 0, 0)으로 다시 넣어요.
  - 업스트림 `get_points`가 `origin - n_voxels / 2 * voxel_size`부터 `voxel_size` 간격으로 점을 만들어요.
  - 박스는 업스트림 `NuScenesDataset.get_ann_info`가 `origin=(0.5, 0.5, 0.5)`로 읽어요.

## 다른 cross-domain 연구의 좌표계 (2026-11-22)

사용자 요청으로 다른 cross-domain·cross-sensor 3D 검출 연구가 좌표계를 어떻게 다루는지 조사했어요. 조사는 에이전트가 했고, "확인"은 제가 원문·코드를 직접 다시 읽은 항목이에요.

| 연구 | 좌표계 | 높이·원점 정렬 | 확인 |
|---|---|---|---|
| **123D** (Dauner 외, arXiv 2605.08084, 2026-05) | 라이브러리는 ISO 8855 차체 좌표계예요. 검출 실험(§4.3)에서 쓴 좌표계는 적혀 있지 않아요. | 축은 통일했어요. §3.1에 "데이터셋마다 ego 원점이 다르다(지면, IMU 위치 등)"고 적혀 있고, 높이는 맞추지 않아요. §4.3은 PETR과 BEVFormer-S(단일 프레임, R50)로 vehicle 클래스, 50 m 이내, 속도·속성을 뺀 NDS를 써요. | 확인 (§3.1, §4.3) |
| **ST3D / ST3D++** (LiDAR UDA) | 지면 원점(Waymo 차량 좌표계) | **맞춰요.** `SHIFT_COOR`로 점과 GT를 옮겨요: KITTI +1.6 m, nuScenes·Lyft +1.8 m. 평가 전에 되돌려요. | 확인 (`secondiou_st3d_car.yaml`의 `SHIFT_COOR: [0.0, 0.0, 1.6]`) |
| **Zheng 외, CoRL 2023** (Cross-Dataset Sensor Alignment, 카메라 3D) | 데이터셋마다 LiDAR 또는 ego 좌표계 | **맞춰요.** 방법이 "camera intrinsic synchronization, extrinsic correction, **ego frame alignment**"이에요. 에이전트에 따르면 보충자료에 "원점 높이가 다르면 BEVDet 3D mAP가 거의 0"이라는 내용이 있어요. | 초록 확인, 보충자료는 미확인 |
| **PD-BEV** (AAAI 2025, `bevdepth/`) | ego | **맞춰요.** CARLA DeepAccident 변환에서 박스와 카메라 위치의 z에 +1.9 m를 더해요(지붕 LiDAR → 지면). | 확인 (`bevdepth/tools/Deepaccident_converter.py:116,153`) |
| **UDGA** (NeurIPS 2024) | LiDAR (`sensor2lidar`) | 공개 코드에서는 z를 옮기지 않아요. | 확인 (`third_party/UDGA/datasets/pipelines/loading.py:1243`) |
| **CoIn3D** (CVPR 2026, 카메라 구성 일반화) | ego (에이전트: 원점이 뒷차축 중심을 지면에 내린 점, 보충자료) | 지면 기준 표현(ground depth 등)을 써요. | 제목·초록만 확인 |
| DG-BEV (CVPR 2023), UniDrive (ICLR 2025), Wang 외 (CVPR 2020) | 각각 ego, 미기재, 데이터셋별 원래 좌표계 (에이전트 보고) | 높이 정렬은 따로 없어요. | 미확인 |

**정리**

- 좌표계를 명시적으로 다루는 연구는 모두 원점을 지면 쪽으로 맞춰요(ST3D, Zheng 외, PD-BEV, CoIn3D). 원래 LiDAR 좌표계를 그대로 쓰는 건 UDGA와 PCCR 공식 LiDAR 계열 모델이에요.
- nuScenes 방식 지표(PCCR, 123D)는 z를 보지 않아요. 그래서 PCCR에서 LiDAR 좌표계 모델이 무너진 건 평가 문제가 아니라, 모델이 샘플링하는 높이 때문이에요.

## 포팅할 때 주의할 점

- **박스 좌표계:** mmdet3d 1.0 형식 `[x,y,z,l,w,h,yaw]`이고 z는 박스 중심이에요(`pccr/lib/data_converter/nuscenes_converter.py:423`). BEVFormer(mmdet3d 0.17.1)로 쓰려면 SECOND 형식으로 변환해야 해요.
- **LiDAR 기준:** LIDAR_TOP은 ego와 회전이 같아요(x 앞). 대신 **높이가 차량마다 달라요**(1.50 / 1.86 / 4.18 / 4.80 m). LiDAR 좌표계로 학습하는 모델(BEVFormer)은 리그 간 z 기준이 어긋나요(위 "박스 z 확인").
- **info 키:** info pkl에 `next`, `frame_idx`, `can_bus` 키가 없어요(`prev`는 정수 인덱스예요). BEVFormer의 `CustomNuScenesDataset`은 그대로 쓰면 KeyError가 나요. CAN bus는 `can_bus/*.json`으로 따로 제공돼요.
- **카메라 자동 선택:** PCCR 공식 BEVDet 설정은 `cams=None`을 쓰는데, 이를 처리하는 파이프라인 코드는 공개되지 않았어요([porting_plan.md](porting_plan.md) 0-A).
- **이미지 크기:** 모든 카메라가 1280×720이에요. 리그 안에서는 크기가 같아요.
- **pkl 로딩:** info pkl은 외부에서 받은 pickle이에요. 학습할 때 mmdet3d가 어차피 로드하지만, 단순히 살펴볼 때는 JSON 테이블(`v1.0-*/`)을 쓰는 걸 권해요.
