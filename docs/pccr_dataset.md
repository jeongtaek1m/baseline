# PCCR 데이터셋

**Plentiful CARLA Camera Rigs (PCCR):** Bader et al., *Understanding Cross-Rig Generalization in Automotive Perception*, ECCV 2026 (arXiv 2606.27554).

| 항목 | 값 |
|---|---|
| HF 데이터셋 | [timb2001/plentiful-carla-camera-rigs](https://huggingface.co/datasets/timb2001/plentiful-carla-camera-rigs), 리비전 `18edf4f4bbc94840d7e4026f6f47b063ba173f04`로 고정 |
| 라이선스 | CC-BY-NC-4.0 (비상업) |
| 벤치마크 코드 | [BaderTim/plentiful-carla-camera-rigs](https://github.com/BaderTim/plentiful-carla-camera-rigs) @ `6798206`, `third_party/pccr/`에 원본 그대로 둠 |
| 서버 위치 | `/data2/PCCR/<RIG>/`, 레포에서는 `data/pccr`(심볼릭 링크)로 접근 |
| 용량 | zip 160.8 GB. 압축을 풀면 리그당 7~18 GB, 전체 약 200 GB |

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
표 지표는 이 중 traffic 2종을 뺀 **7-class mAP**예요([evaluation_protocol.md](evaluation_protocol.md)).

## 포팅할 때 주의할 점

- **박스 좌표계:** mmdet3d 1.0 형식 `[x,y,z,l,w,h,yaw]`예요(`pccr/lib/data_converter/nuscenes_converter.py:421`). BEVFormer(mmdet3d 0.17.1)로 쓰려면 SECOND 형식으로 변환해야 해요.
- **LiDAR 기준:** LIDAR_TOP은 ego와 회전이 같아요(x 앞). 대신 **높이가 차량마다 달라요**(1.50 / 1.86 / 4.18 / 4.80 m). LiDAR 좌표계로 학습하는 모델(BEVFormer)은 리그 간 z 기준이 어긋나요.
- **info 키:** info pkl에 `next`, `frame_idx`, `can_bus` 키가 없어요(`prev`는 정수 인덱스예요). BEVFormer의 `CustomNuScenesDataset`은 그대로 쓰면 KeyError가 나요. CAN bus는 `can_bus/*.json`으로 따로 제공돼요.
- **카메라 자동 선택:** PCCR 공식 BEVDet 설정은 `cams=None`을 쓰는데, 이를 처리하는 파이프라인 코드는 공개되지 않았어요([porting_plan.md](porting_plan.md) 0-A).
- **이미지 크기:** 모든 카메라가 1280×720이에요. 리그 안에서는 크기가 같아요.
- **pkl 로딩:** info pkl은 외부에서 받은 pickle이에요. 학습할 때 mmdet3d가 어차피 로드하지만, 단순히 살펴볼 때는 JSON 테이블(`v1.0-*/`)을 쓰는 걸 권해요.
