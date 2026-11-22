# Gen2E: vanilla BEVDepth (single frame) on PCCR. Train on R1, test on every rig's test split.
#
# Base: PCCR's official BEVDet config (third_party/pccr/models/BEVDet/bevdet-r50-pccr.py).
# That config targets BEVDet dev3.0, so its pipeline is translated to the dev2.0 pipeline of
# this tree (LoadAnnotations + BEVAug -> LoadAnnotationsBEVDepth). On top of it come the
# upstream BEVDepth parts (diff of upstream bevdet-r50.py vs bevdet4d-r50-depth-cbgs.py):
# LSSViewTransformerBEVDepth with DepthNet(use_dcn=False), LiDAR depth supervision through
# PointToMultiViewDepth, and the BEVDepth4D detector with num_adj=0 (one frame, no
# pre_process).
#
# Differences from the PCCR BEVDet config:
#   * 7 classes: traffic_sign and traffic_light are dropped (user decision 2026-11-22).
#   * The BEVDepth additions above.
#   * Classes that PD-BEV edited in place are taken from verbatim upstream copies
#     (*Upstream, see tools/check_bevdepth_upstream.py).
#   * PrepareImageInputsPCCR implements cams=None (missing from the PCCR repo); ego_cam=None.
#   * CenterHeadNoVel: upstream dev2.0 CenterHead whose loss also works without a 'vel' head
#     (PCCR trains without velocity; dev2.0's loss hard-codes it, dev3.0's does not).
# Everything else (input size, augmentation, grid, head, optimizer, schedule, EMA, batch
# 2 GPUs x 4) follows the PCCR config.

custom_imports = dict(
    imports=[
        'mmdet3d.datasets.pccr_dataset',
        'mmdet3d.datasets.pipelines.loading_upstream',
        'mmdet3d.datasets.pipelines.loading_pccr',
        'mmdet3d.models.dense_heads.centerpoint_head_upstream',
        'mmdet3d.models.dense_heads.centerpoint_head_pccr',
    ],
    allow_failed_imports=False)

point_cloud_range = [-51.2, -51.2, -6.0, 51.2, 51.2, 6.0]
class_names = ['car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult', 'child']
pccr_class_range = dict(
    car=50, truck=50, bus=50, bicycle=40, motorcycle=40, adult=40, child=40)

dataset_type = 'PCCRDataset'
# Relative to bevdepth/ (the working directory of tools/train.py and tools/test.py).
# The repository links data/pccr -> /data2/PCCR. tools/bevdepth_*.sh override it per rig.
data_root = '../data/pccr/R1/'
file_client_args = dict(backend='disk')
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

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
    depth=[1.0, 60.0, 1.0])

voxel_size = [0.1, 0.1, 0.2]
numC_Trans = 64

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

model = dict(
    type='BEVDepth4D',
    align_after_view_transfromation=False,
    num_adj=0,
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVDepth',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=numC_Trans,
        depthnet_cfg=dict(use_dcn=False),
        downsample=16),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=256),
    pts_bbox_head=dict(
        type='CenterHeadNoVel',
        in_channels=256,
        tasks=[dict(num_class=len(class_names), class_names=class_names)],
        common_heads=dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=7),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            pre_max_size=1000,
            post_max_size=500,
            nms_type=['rotate'],
            nms_thr=[0.2],
            nms_rescale_factor=[[1.0] * len(class_names)])))

train_pipeline = [
    dict(
        type='PrepareImageInputsPCCR',
        is_train=True,
        data_config=data_config,
        ego_cam=None),
    dict(
        type='LoadAnnotationsBEVDepthUpstream',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='PointToMultiViewDepthUpstream',
        downsample=1,
        grid_config=grid_config),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_depth'])
]

test_pipeline = [
    dict(type='PrepareImageInputsPCCR', data_config=data_config, ego_cam=None),
    dict(
        type='LoadAnnotationsBEVDepthUpstream',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
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
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
]

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    img_info_prototype='bevdet',
    with_velocity=False,
    discard_velocity_eval=True,
    class_range=pccr_class_range,
    ego_cam='auto')

test_data_config = dict(
    data_root=data_root,
    pipeline=test_pipeline,
    ann_file=data_root + 'R1_infos_test.pkl',
    test_mode=True,
    box_type_3d='LiDAR')

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'R1_infos_train.pkl',
        pipeline=train_pipeline,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR'),
    val=dict(
        data_root=data_root,
        pipeline=test_pipeline,
        ann_file=data_root + 'R1_infos_val.pkl',
        test_mode=True,
        box_type_3d='LiDAR'),
    test=test_data_config)

for key in ['train', 'val', 'test']:
    data[key].update(share_data_config)

evaluation = dict(interval=1)
checkpoint_config = dict(interval=4)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'

optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-07)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[24])
runner = dict(type='EpochBasedRunner', max_epochs=24)

custom_hooks = [
    dict(type='MEGVIIEMAHook', init_updates=10560, priority='NORMAL'),
]
