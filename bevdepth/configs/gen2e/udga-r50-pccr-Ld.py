# Gen2E: BEVDepth + UDGA, variant UDGA-Ld = the public config's loss set: LiDAR depth on
# (bevssl3sup_..._pc400_L1_extaug.py: loss_depth_weight 100, dbg=False) + L_ov + L_p (public
# code, no gradient) + ext aug.
#
# Decision 2026-11-23 (docs/decisions.md): runs in addition to the main run (./udga-r50-pccr.py)
# and UDGA-Lp. The LiDAR depth loss is the vanilla BEVDepth one (dev2.0 BCE on the softmax,
# GT min-pooled per 16x16 cell, weight 3), as for every BEVDepth-family row; its targets are
# moved into the virtual-depth bins (v = d * k, DG-BEV eq. 3) so they match the lift
# (UDGA-2: the official target d / k does not). Everything else as ./udga-r50-pccr.py.
#
# Train: bash tools/bevdepth_train.sh configs/gen2e/udga-r50-pccr-Ld.py 2,3 29523
# Test:  bash tools/bevdepth_test_rigs.sh bevdepth/work_dirs/udga-r50-pccr-Ld/R1 2,3

_base_ = ['./udga-r50-pccr.py']

class_names = ['car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult', 'child']
file_client_args = dict(backend='disk')

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
    lidar_depth_loss=True,
    img_view_transformer=dict(loss_depth_weight=3.0))

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
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    # metric depth labels in [1, 90) m; the loss converts them to virtual depth per camera
    dict(type='PointToMultiViewDepthUpstream', downsample=1, grid_config=grid_config),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_depth'])
]

data = dict(train=dict(pipeline=train_pipeline))
