# Gen2E: BEVDepth + UDGA (domain-generalisation stage) on PCCR. Train on R1, test on every rig.
#
# Base: our vanilla BEVDepth config (./bevdepth-r50-pccr.py): same backbone, image neck (256),
# DepthNet, BEV channels (64), grid x/y/z, head (7-class CenterHeadNoVel), image/BEV
# augmentation, optimizer, schedule (24 epochs), EMA and batch (2 GPUs x 4).
# UDGA (NeurIPS 2024, third_party/UDGA config bevssl3sup_lyf_to_nus_cls1_d60_pc400_L1_extaug.py)
# re-implemented on top, as decided on 2026-11-23 (docs/decisions.md):
#   * No LiDAR depth loss (paper table 4 last row / eq. 7): no LiDAR loading at all.
#   * Multi-view overlap depth constraint L_ov: depth logits of camera i moved to the
#     corresponding frustum cells of camera j, SmoothL1 (beta 1) on the corresponding cells
#     only, weight 1 (official ssl_weight). Pairs: every ordered camera pair with overlap,
#     computed per sample from the calibration (R1: 16 neighbour pairs).
#   * Photometric term L_p as the public code (weight 2.0, no gradient: logged only).
#   * DG-BEV virtual depth bins (pixel_const 400, bins [1, 90) step 1 as the official config).
#   * Ext aug: +{-90, 90, 180} deg BDA rotation with probability 0.5.
#   * Official-code bugs fixed as the paper intends: docs/bug_ledger.md (UDGA-1 ... 9).
#   * Spatial pairs only, single frame (as the public code).
#
# Train: bash tools/bevdepth_train.sh configs/gen2e/udga-r50-pccr.py 2,3 29520
# Test:  bash tools/bevdepth_test_rigs.sh bevdepth/work_dirs/udga-r50-pccr/R1 2,3

_base_ = ['./bevdepth-r50-pccr.py']

custom_imports = dict(
    imports=[
        'mmdet3d.datasets.pccr_dataset',
        'mmdet3d.datasets.pipelines.loading_upstream',
        'mmdet3d.datasets.pipelines.loading_pccr',
        'mmdet3d.datasets.pipelines.loading_udga',
        'mmdet3d.models.dense_heads.centerpoint_head_upstream',
        'mmdet3d.models.dense_heads.centerpoint_head_pccr',
        'mmdet3d.models.necks.view_transformer_udga',
        'mmdet3d.models.detectors.bevdepth_udga',
    ],
    allow_failed_imports=False)

class_names = ['car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult', 'child']

data_config = dict(
    cams=None,
    Ncams=None,
    input_size=(384, 704),
    src_size=(720, 1280),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.0)

# depth bins are UDGA's virtual depths (official dbound [1, 90, 1]); x/y/z as vanilla
grid_config = dict(
    x=[-51.2, 51.2, 0.8],
    y=[-51.2, 51.2, 0.8],
    z=[-6, 6, 12],
    depth=[1.0, 90.0, 1.0])

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

model = dict(
    type='BEVDepthUDGA',
    overlap_loss_weight=1.0,
    overlap_loss_beta=1.0,
    photo_loss_weight=2.0,
    img_view_transformer=dict(
        type='LSSViewTransformerBEVDepthUDGA',
        pixel_const=400.0,
        grid_config=grid_config))

train_pipeline = [
    dict(
        type='PrepareImageInputsPCCR',
        is_train=True,
        data_config=data_config,
        ego_cam=None),
    dict(
        type='LoadAnnotationsBEVDepthExtAug',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        ext_aug_prob=0.5,
        ext_aug_angles=(-90.0, 90.0, 180.0)),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='PrepareImageInputsPCCR', data_config=data_config, ego_cam=None),
    dict(
        type='LoadAnnotationsBEVDepthUpstream',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img_inputs'])
        ])
]

data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
