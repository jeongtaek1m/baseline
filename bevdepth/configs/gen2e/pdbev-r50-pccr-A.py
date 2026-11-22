# Gen2E: BEVDepth + PD-BEV (domain generalisation) on PCCR, perturbation ranges A (paper).
#
# Base: our vanilla BEVDepth config (./bevdepth-r50-pccr.py): same backbone, image neck (256),
# BEV channels (64), grid, head (7-class CenterHeadNoVel), image/BEV augmentation, optimizer,
# schedule, EMA and batch (2 GPUs x 4). Decisions 2026-11-22 (docs/decisions.md):
#   * PD-BEV components on top of it, following the official DG config
#     (configs/PDBEV/pdbev-r50-cbgs-NUS2X-dg.py) and the paper: PCBEV_DG, virtual depth
#     (depth bins [1, 100]), IFV with 4 height bins, real + virtual view rendering, 2D heads on
#     the images (L_ps) and on the rendered views (L_render), box-centre depth loss (L_pg).
#     PD-BEV does not use LiDAR depth (paper), so the LiDAR loading steps are dropped.
#   * No car-only BEV auxiliary loss (not in the paper's eq. 8).
#   * Official code bugs fixed as the paper intends: docs/bug_ledger.md (PDBEV-1 ... 18).
#   * Loss weights and normalisation as the official code (paper: all lambda_s = 1).
#   * Perturbation of the virtual cameras, A = official config values: pitch/roll +-0.04 rad,
#     yaw +-0.4 rad, xyz +-2 m. (B: ./pdbev-r50-pccr-B.py.)
#   * No CBGS (as vanilla and PCCR BEVDet).
#
# Train: bash tools/bevdepth_train.sh configs/gen2e/pdbev-r50-pccr-A.py 0,1
# Test:  bash tools/bevdepth_test_rigs.sh bevdepth/work_dirs/pdbev-r50-pccr-A/R1 0,1

_base_ = ['./bevdepth-r50-pccr.py']

class_names = ['car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult', 'child']
point_cloud_range = [-51.2, -51.2, -6.0, 51.2, 51.2, 6.0]
voxel_size = [0.1, 0.1, 0.2]
numC_Trans = 64          # vanilla BEV channels (official PD-BEV: 80)
neck_channels = 256      # vanilla image neck (official PD-BEV: 512)

data_config = dict(
    cams=None,
    Ncams=None,
    input_size=(384, 704),
    src_size=(720, 1280),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.0,
    # PD-BEV virtual-camera perturbation, A = official config (pdbev-r50-cbgs-NUS2X-dg.py:34-39)
    re_ratio=False,
    pitch_aug=(-0.04, 0.04),
    yaw_aug=(-0.4, 0.4),
    roll_aug=(-0.04, 0.04),
    extri_x_aug=(-2.0, 2.0),
    extri_y_aug=(-2.0, 2.0),
    extri_z_aug=(-2.0, 2.0))

# virtual depth bins of PD-BEV (DG config); x/y/z as vanilla
grid_config = dict(
    x=[-51.2, 51.2, 0.8],
    y=[-51.2, 51.2, 0.8],
    z=[-6, 6, 12],
    depth=[1.0, 100.0, 1.0])

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

aux_channels = len(class_names) + len(class_names) * 6

model = dict(
    _delete_=True,
    type='PCBEV_DG',
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
    img_aug=dict(
        type='Img_Aux',
        numC_input=neck_channels,
        class_name=class_names,
        upsample=[1, 2, 1],
        num_channels=[256, 128, aux_channels]),
    bev_img_aux=dict(
        type='Img_Aux_Dy',
        numC_input=numC_Trans,
        class_name=class_names,
        num_layer=[2, 1, 1],
        upsample=[2, 1, 1],
        num_channels=[256, 128, aux_channels]),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=neck_channels,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformer_pc',
        BEV_Aux=dict(
            type='BEV_Aux',
            class_name=class_names,
            numC_input=256,
            num_layer=[2, 2, 2],
            height_num=4,
            num_channels=[numC_Trans, numC_Trans, numC_Trans + 4]),
        downsample_from_ann=2,
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=neck_channels,
        depthnet_cfg=dict(use_dcn=False),
        out_channels=numC_Trans,
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
    dict(type='PrepareImageInputs_UDA', is_train=True, data_config=data_config, ego_cam=None),
    dict(type='LoadAnnotationsBEVDepth', bda_aug_conf=bda_aug_conf, classes=class_names),
    # 2D targets for the real and virtual views (objects filtered as below: PDBEV-5)
    dict(type='Load3DBoxesHeatmap', classes=class_names, downsample_feature=8,
         point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(type='ObjectRangeFilter', point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d',
              'heatmaps_2d', 'ann_maps_2d', 'heatmap_masks_2d',
              'heatmaps_2d_aug', 'ann_maps_2d_aug', 'heatmap_masks_2d_aug',
              'bev_aug'])
]

test_pipeline = [
    dict(type='PrepareImageInputs_UDA', data_config=data_config, ego_cam=None),
    dict(type='LoadAnnotationsBEVDepth', bda_aug_conf=bda_aug_conf, classes=class_names, is_train=False),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
            dict(type='Collect3D', keys=['img_inputs'])
        ])
]

data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
