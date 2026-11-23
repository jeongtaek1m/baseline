# 결정 사항 기록

포팅과 실험 설정에 영향을 주는 결정을 날짜순으로 남겨요. "확인 필요" 항목은 아직 확정되지 않은 것들이에요.

## 확정

| 날짜 | 결정 | 근거·비고 |
|---|---|---|
| 2026-11-21 | 데이터셋은 **PCCR** (Plentiful CARLA Camera Rigs, ECCV 2026)를 쓴다. HF 리비전 `18edf4f`로 고정하고 `/data2/PCCR`에 둔다. | 표의 컬럼(R1, R1-c6, …, R9)이 PCCR의 14개 리그와 같아요. 루트 디스크(`/`) 여유가 부족해서 `/data2`에 둬요. |
| 2026-11-21 | 사용자가 말한 "ugbev"는 **UDGA** (NeurIPS 2024, SAITPublic/UDGA)다. | 사용자 확인. |
| 2026-11-21 | 이번 포팅 범위는 BEVDepth(+PD-BEV, +UDGA)와 BEVFormer-S(+PD-BEV)다. vanilla BEVDepth 행도 함께 채운다. | 표에서 BEVDepth 행 자체가 비어 있어요. UniDrive와 CoIn3D는 보류해요. |
| 2026-11-21 | BEVFormer-S는 **bevformer_small 구조 + 단일 프레임**(temporal queue 없음)으로 한다. | 사용자 확인. 공식 small 설정은 temporal queue를 쓰기 때문에 끄는 수정이 필요해요. |
| 2026-11-21 | **모델당 GPU 2장**을 쓴다. 4090 4장에서 두 모델을 동시에 학습한다. | 사용자 지시. PCCR 공식 BEVDet도 GPU 2장 × 4샘플이었어요. |
| 2026-11-21 | 레포는 공저자 코드 검토용 **private 모노레포**로 한다. 원본 스냅샷을 먼저 커밋하고, 변경은 커밋을 나눠서 넣는다. | 사용자: "배포가 아니라 공저자와 코드가 틀린 데 없는지 비교하려고 올린다". |
| 2026-11-21 | 문서는 한국어로 쓴다. | 사용자 응답이 없어서 기본값으로 정했어요. 영어로 바꾸기 쉬워요. |
| 2026-11-22 | **BEVDepth base는 PD-BEV 레포를 그대로 쓴다** (BEVDet dev2.0 @2ae6a02에서 갈라진 트리). | 사용자가 재확인했어요. 깨끗한 원본 BEVDet(@2ae6a02 또는 dev3.0 @26144be)도 선택지로 제시했지만 PD-BEV 트리를 골랐어요. 주의사항은 바로 아래에 있어요. |
| 2026-11-22 | **학습도 7개 클래스**(car, truck, bus, bicycle, motorcycle, adult, child)로 한다. traffic_sign과 traffic_light는 학습에서 빼고, 보고는 7-class mAP(test split)로 한다. | 사용자 지시("학습할 때 traffic light랑 빼고"). PCCR 공식 † 행은 9개 클래스로 학습했으니, 우리 행과 학습 클래스 수가 달라요. |
| 2026-11-22 | BEVDepth 계열 학습 설정은 **PCCR BEVDet 기준**(GPU 2장 × 4샘플, AdamW 2e-4, 24 epoch, 384×704)으로 한다. 방법별 추가 손실 가중치만 원 논문 값을 쓴다. | 사용자 승인. BEVFormer-S 학습 설정은 아래에 따로 있어요. |
| 2026-11-22 | ~~BEVFormer-S는 LiDAR 좌표계로 학습한다 (선택지 A).~~ 같은 날 아래 결정으로 바뀌었어요. | 사용자 결정("a ㄱㄱ")이었어요. 이후 PCCR 공식 모델 조사를 보고 사용자가 다시 정했어요. |
| 2026-11-22 | **BEVFormer-S는 차량 고정 좌표계로 학습한다.** 축은 ego와 같고, 원점은 모든 차량에서 R1의 LiDAR 장착 위치(ego 기준 x −0.46, z +1.86 m)예요. 설정값은 `pccr_ref2ego = [-0.46, 0, 1.86]`이에요. | 사용자: "lidar로 하면 차종마다 달라져서 ego로". LiDAR 좌표계는 차량마다 지면 높이가 1.50~4.80 m로 달라지고, PCCR에서 LiDAR 좌표계로 높이를 여러 층 샘플링한 PETR·Fast-BEV는 R6·R9에서 0.0이었어요. 이 원점을 쓰면 etron 리그 9개는 원래 LiDAR 좌표계와 수치까지 같고(확인함), 다른 차량은 지면이 z = −1.86으로 맞춰져요. BEVFormer z 범위([−5, 3])도 그대로예요. 순수 ego(지면 z=0)와는 평행이동만 다르고, 설정값을 `[0, 0, 0]`으로 바꾸고 z 범위를 1.86 m 올리면 돼요. 다른 cross-domain 연구도 원점을 지면 쪽으로 맞춰요([pccr_dataset.md](pccr_dataset.md) "다른 cross-domain 연구의 좌표계"). |
| 2026-11-22 | **BEVFormer-S 학습 설정:** 배치 8(GPU 2장 × 1샘플 × gradient 누적 4회), AdamW lr 2e-4, 24 epoch, 입력 1280×720(배율 1.0), 백본은 ImageNet 가중치(detectron2 caffe R101, `resnet101_msra-6cc46731.pth`), CAN bus 끔. | 사용자 승인(네 가지 모두 추천안). 공식 small과 PCCR 모델들의 배치 8에 맞췄어요. FCOS3D(nuScenes 학습) 가중치는 쓰지 않아요. 속도 head 없음(PCCR PETR·BEVDet과 같음)과 `cams_embeds` 끔(카메라 수가 리그마다 달라서)은 강제되는 선택이라 따로 묻지 않았어요. |
| 2026-11-22 | ~~PD-BEV head: `bev_loss`가 7클래스 head의 car 채널을 쓰게 고친다.~~ 같은 날 아래 결정으로 바뀌었어요. | 사용자 결정(추천안)이었어요. 이후 이 손실이 논문에 없다는 걸 확인했어요. |
| 2026-11-22 | **car 전용 BEV 보조 손실(`bev_loss`)은 뺀다.** head는 vanilla BEVDepth와 같은 7클래스 단일 task예요(`CenterHeadNoVel`). | 사용자 결정(추천안). 논문 식 8에 이 손실이 없어요. 공식 학습에서도 PDBEV-2(eq(0))와 PDBEV-17(ReLU 바닥) 때문에 사실상 동작하지 않았어요. |
| 2026-11-22 | **PD-BEV 2D·렌더 head의 손실 정규화(PDBEV-13·14)는 공식 그대로 둔다.** ReLU 바닥(PDBEV-17)과 IFV sigmoid(PDBEV-12)는 고친다. | 사용자 결정(추천안). 논문에는 "CenterPoint를 따른다"만 있고 정규화는 없어요. 바꾸면 손실 크기가 수백 배 달라져서, 논문에 없는 가중치를 새로 정해야 해요. |
| 2026-11-22 | **BEVDepth + PD-BEV의 기본 구조는 vanilla BEVDepth와 같게 한다:** 채널 64, 이미지 neck 256, BEV z [−6, 6]. 여기에 PD-BEV 구성요소를 더해요: virtual depth [1, 100], IFV 4칸, 렌더링, 2D·렌더 head. | 사용자 결정(추천안). vanilla 대비 차이가 PD-BEV 때문인지 분명해요. 학습 설정 원칙에 따라 CBGS는 쓰지 않아요(vanilla·PCCR BEVDet과 같음). PD-BEV는 LiDAR depth를 쓰지 않으니(논문) LiDAR 로딩 단계도 빼요. |
| 2026-11-22 | **PD-BEV·UDGA 공식 코드 버그는 논문 의도대로 고친다.** 고친 내용은 모두 문서에 남기고, 표에는 수정 재구현이라고 밝힌다. | 사용자 결정. 추천안은 공식 동작 유지였어요. 버그 목록은 [porting_plan.md](porting_plan.md) 2장, 3장에 있어요. |
| 2026-11-22 | **PD-BEV 섭동 범위는 두 가지 다 실험한다.** PD-BEV-A는 공식 설정 값이에요: pitch·roll ±0.04 rad, yaw ±0.4 rad, xyz ±2 m. yaw는 논문 본문에 0.2로 적혀 있지만 논문 그림 9와 표 1이 0.4와 맞아서 0.4를 써요. PD-BEV-B는 PCCR 리그에 맞춘 값이에요: yaw ±22.5°, pitch ±20°, roll ±0.04 rad, x −2.9~+5.3 m, y −0.6~+0.8 m, z −0.5~+2.7 m. | 사용자 결정("둘다 실험 논문은 A, pccr rig 는 B", A yaw 0.4와 B 범위는 추천안). B는 PCCR 테스트 카메라마다 시야 방향이 가장 가까운 R1 카메라와 비교한 최대 변화량이에요(리그 정의 JSON에서 계산). 테스트 리그 정보를 쓰므로 표에서 따로 표시해야 해요. 외부 파라미터만 섭동하고, FOV 변화는 virtual depth가 처리해요. pitch 범위가 실제로 카메라를 기울이게 하려면 PDBEV-10을 먼저 고쳐야 해요([bug_ledger.md](bug_ledger.md)). |
| 2026-11-23 | **UDGA는 LiDAR depth 지도학습(L_d) 없이 학습한다.** 손실은 L_det + L_ov + L_p예요. | 사용자 결정(추천안). 논문 표 4의 최종 행(Lyft→nuScenes 0.421 = 표 1 'Ours')과 식 7에 L_d가 없어요. 공개 config는 L_d를 켜 뒀어요(가중치 100). 표 4에서 L_d + L_ov + L_p(0.396)는 L_ov + L_p(0.407)보다 낮았어요. PD-BEV 행처럼 LiDAR 로딩 단계를 빼요. |
| 2026-11-23 | **UDGA는 DG-BEV식 초점거리 정규화 depth를 쓴다:** `pixel_const=400`, virtual depth bin [1, 90, 1](D=89). | 사용자 결정(추천안). 공개 config 값이에요. 논문 방법 설명에는 없지만 'DG-BEV를 따라 학습'이라고 쓰고, 수치(L_ov만 0.403 vs DG-BEV 0.374)도 이걸 포함한 것으로 보여요. metric depth = virtual / k, k = 400·√(f_x'⁻² + f_y'⁻²)(DG-BEV 식 3). R1은 k = 1.02예요. 120° 카메라(k = 2.78)에서는 최대 32.3 m까지 표현돼요. lift와 L_ov 재투영이 같은 변환을 써요(UDGA-2·3). PD-BEV 행이 PD-BEV 자체 virtual depth [1, 100]을 쓴 것과 같은 원칙이에요. |
| 2026-11-23 | **UDGA ext aug를 넣는다.** 학습 샘플마다 확률 0.5로 BDA 회전에 {−90°, 90°, 180°} 중 하나를 더해요. | 사용자 결정(추천안). 논문 최종 행과 공개 config 모두 포함해요(0.407 → 0.421). 공식은 학습셋을 통째로 복제하고 복제본마다 각도를 하나 고정해서 epoch당 반복이 두 배예요. 여기서는 같은 50:50 비율을 샘플링으로 만들어서 반복 수를 vanilla와 같게 둬요. BDA로 하니 공식의 LiDAR 점 좌표계 버그(UDGA-5)는 생기지 않아요. 원래 목적인 Lyft·nuScenes LiDAR 축 방향 차이는 ego 좌표계인 PCCR에는 없어요. |
| 2026-11-23 | **UDGA L_ov·L_p는 같은 시각의 공간 인접 카메라 쌍만 쓴다.** 쌍은 하드코딩하지 않고, 캘리브레이션과 증강 행렬로 매 샘플 계산한 frustum 대응이 있는 모든 방향 쌍이에요. | 사용자 결정(추천안). 공개 코드와 같아요. 논문 본문은 시간 인접 뷰도 언급해요. R1(8대)은 이웃끼리 방향 쌍 16개이고, 픽셀의 24~36%가 이웃과 겹쳐요. |
| 2026-11-23 | **UDGA L_ov는 대응이 있는 frustum 칸만 비교한다.** 카메라 i의 depth logit을 카메라 j의 frustum 칸으로 옮기고, 같은 칸에 온 값은 평균해서 j의 logit과 SmoothL1(β=1, 가중치 1)로 비교해요. | 사용자 결정(추천안). logit·SmoothL1·가중치 1은 공개 config(`ssl_loss='l1_loss'`, `ssl_weight=1.0`) 그대로예요. 공개 코드는 겹친 픽셀의 모든 bin을 비교해서, 값이 옮겨지지 않은 bin은 0과 비교해요(UDGA-9). 논문 식 3은 '대응 depth' 사이 거리예요. |
| 2026-11-23 | **UDGA L_p는 일단 공개 코드대로 둔다.** argmax depth와 정수 인덱싱으로 계산하고 가중치 2.0으로 로그에 남기지만, 그래디언트는 0이에요. | 사용자 결정("일단 공개 코드 대로"). 추천안은 논문 식 4대로 그래디언트가 흐르게 다시 만드는 것이었어요. 공개 코드와 학습 동작이 같아요. 쌍 계산(UDGA-7)과 배치 계산(UDGA-8)만 PCCR에서 돌도록 바꿨어요. |
| 2026-11-23 | 표의 BEVDepth 행을 우리 값으로 바꿀지는 **사용자가 직접 다시 검증한 뒤 정한다.** | 우리 BEVDepth는 R1 30.4로, 사용자 표의 BEVDepth [42](36.9)보다 6.4점 낮아요. R1 계열 리그는 모두 5.5~6.9점 낮고, 먼 리그는 비슷해요(하락률 53.5% vs 55.0%). 좌표계는 같은 ego예요. `tools/check_bevdepth_pccr.py`로 평가 체인과 학습 기하를 확인했어요. |

### BEVDepth base = PD-BEV 트리: 반드시 지킬 것

PD-BEV는 원본 BEVDet 파일 16개를 직접 고쳤어요. 그래서 그 트리에서 config만 바꿔 "vanilla BEVDepth"를 돌리면 표준 BEVDepth가 아니에요.

- **`PointToMultiViewDepth`(loading.py):** depth 라벨을 초점거리 450 px 기준 virtual depth로 바꾸고, 범위 필터를 ±100으로 바꿨어요. `img_inputs`에 8번째 원소(`act_intrins`)도 요구해요.
- **PD-BEV 자체 BEVDepth 경로:** `torch.max` 관련 버그로 LiDAR depth 지도학습이 사실상 꺼져 있어요(검증 에이전트 확인).
- **`CenterHead`:** 818줄이 바뀌었어요(PD-BEV 손실 추가, yaw weight 등).

그래서 **vanilla BEVDepth는 원본 2ae6a02 클래스를 별도 이름으로 등록해서 그 경로로만** 돌려요. PD-BEV 설정은 PD-BEV 클래스를 쓰고요. 이 복원 작업은 별도 커밋으로 만들어서 공저자가 검토할 수 있게 해요.

## 확인 필요 (사용자 답변 대기)

| 항목 | 제안(기본값) | 왜 중요한가 |
|---|---|---|
| 기존 BEVFormer-S(PCCR) 코드 | 답이 없어서 새로 만들었어요(2026-11-22). 공식 BEVFormer 위에 PCCR 어댑터를 붙였어요. 기존 코드가 있으면 알려 주세요. 그 방식과 비교해서 맞출게요. | 표의 기존 BEVFormer-S 행(R1 21.1, R1-c6 19.6, R1-c10 26.2 …)을 어떻게 만들었는지 몰라요. 좌표계, `cams_embeds`, 해상도, 학습 설정이 다르면 우리 행과 수치가 달라질 수 있어요. 기존 행의 R6·R9가 0 근처면 LiDAR 좌표계로 돌렸다는 뜻이에요. |
