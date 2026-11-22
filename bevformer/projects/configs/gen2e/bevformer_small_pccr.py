# Gen2E: BEVFormer-S on PCCR, trained on R1.
#
# Copy of projects/configs/bevformer/bevformer_small.py; every change is marked "Gen2E".
# Decisions (docs/decisions.md):
#   2026-11-21  bevformer_small architecture, single frame (no temporal queue)
#   2026-11-22  7 classes (traffic_sign / traffic_light excluded)
#   2026-11-22  vehicle-fixed reference frame instead of each rig's LiDAR frame (see pccr_ref2ego)
#   2026-11-22  batch 8 = 2 GPUs x 1 sample x 4 accumulated iterations, lr 2e-4, 24 epochs
#   2026-11-22  1280x720 input, ImageNet backbone initialisation, CAN bus off
#
# Train:  bash tools/bevformer_train.sh projects/configs/gen2e/bevformer_small_pccr.py 2,3
# Test:   bash tools/bevformer_test_rigs.sh bevformer/work_dirs/bevformer_small_pccr/R1 2,3

# BEvFormer-small consumes at lease 10500M GPU memory
# compared to bevformer_base, bevformer_small has
# smaller BEV: 200*200 -> 150*150
# less encoder layers: 6 -> 3
# smaller input size: 1600*900 -> (1600*900)*0.8
# multi-scale feautres -> single scale features (C5)
# with_cp of backbone = True

_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
# Gen2E: the PCCR dataset is imported here instead of being added to the plugin's datasets/__init__.py.
custom_imports = dict(
    imports=['projects.mmdet3d_plugin.datasets.pccr_dataset'],
    allow_failed_imports=False)

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]


img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
# Gen2E: PCCR, 7 classes (traffic_sign and traffic_light are not trained or evaluated).
class_names = ['car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult', 'child']
# Gen2E: PCCR evaluation ranges, as in PCCR's BEVDet config and our BEVDepth config.
pccr_class_range = dict(
    car=50, truck=50, bus=50, bicycle=40, motorcycle=40, adult=40, child=40)
# Gen2E: reference frame = ego axes with the origin at R1's LIDAR_TOP mount (x -0.46 m, z +1.86 m
# in the ego frame, whose origin is on the ground). Every vehicle then has its ground at z = -1.86,
# and the etron rigs (R1, R1-*, R2, R3, R7) see exactly upstream's LiDAR frame. With each rig's
# own LiDAR frame (None) the ground would move with the LiDAR height: 1.50 m (audi.tt), 1.86 m
# (etron), 4.18 m (HGV), 4.80 m (firetruck). [0, 0, 0] would be the plain ego frame (then shift the
# pc_range z window by +1.86 m). docs/pccr_dataset.md, docs/decisions.md (2026-11-22).
pccr_ref2ego = [-0.46, 0.0, 1.86]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 1
bev_h_ = 150
bev_w_ = 150
queue_length = 1  # Gen2E: single frame (was 3)

model = dict(
    type='BEVFormer',
    use_grid_mask=True,
    video_test_mode=False,  # Gen2E: single frame (was True)
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        with_cp=True, # using checkpoint to save GPU memory
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True),
        # Gen2E: ImageNet initialisation (detectron2 caffe R101, the initialisation FCOS3D started
        # from) instead of load_from the nuScenes-trained FCOS3D checkpoint. DCN offsets start at 0.
        # Fetch with tools/download_bevformer_pretrained.sh.
        init_cfg=dict(type='Pretrained', checkpoint='ckpts/resnet101_msra-6cc46731.pth')),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='BEVFormerHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_classes=7,  # Gen2E: was 10
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        # Gen2E: no velocity (was code_size 10 with weights 0.2 for vx, vy). PCCR's models are
        # trained without velocity (PETR code_size=8, BEVDet with_velocity=False).
        code_size=8,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='PerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=False,  # Gen2E: CAN bus off (was True); the dataset also passes zeros
            # Gen2E: no per-camera embedding (was True by default). Rigs have 4-10 cameras in
            # different orders, so an embedding per camera index cannot transfer between rigs.
            use_cams_embeds=False,
            embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=3,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],

                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=7),  # Gen2E: was 10
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range))))

dataset_type = 'PCCRBEVFormerDataset'  # Gen2E: was CustomNuScenesDataset
# Gen2E: relative to bevformer/ (the working directory of tools/train.py and tools/test.py).
# The repository links data/pccr -> /data2/PCCR. tools/bevformer_test_rigs.sh overrides it per rig.
data_root = '../data/pccr/R1/'
file_client_args = dict(backend='disk')


train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    # Gen2E: scale 1.0 (was 0.8). PCCR cameras are 1280x720, the input size small gets from
    # nuScenes 1600x900 x 0.8.
    dict(type='RandomScaleImageMultiViewImage', scales=[1.0]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    # dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1280, 720),  # Gen2E: was (1600, 900); not used by the transforms below
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[1.0]),  # Gen2E: was 0.8
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'R1_infos_train.pkl',  # Gen2E: PCCR info file
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        class_range=pccr_class_range,  # Gen2E
        ref2ego_translation=pccr_ref2ego,  # Gen2E
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file=data_root + 'R1_infos_val.pkl',  # Gen2E: PCCR info file
             pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),
             class_range=pccr_class_range,  # Gen2E
             ref2ego_translation=pccr_ref2ego,  # Gen2E
             classes=class_names, modality=input_modality, samples_per_gpu=1),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file=data_root + 'R1_infos_test.pkl',  # Gen2E: PCCR info file (test split)
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              class_range=pccr_class_range,  # Gen2E
              ref2ego_translation=pccr_ref2ego,  # Gen2E
              classes=class_names, modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

# Gen2E: total batch 8 as the official small (8 GPUs x 1 sample) and PCCR's models (2 GPUs x 4):
# 2 GPUs x 1 sample x 4 accumulated iterations (was a plain optimizer hook with 8 GPUs).
optimizer_config = dict(
    type='GradientCumulativeOptimizerHook',
    cumulative_iters=4,
    grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=2000,  # Gen2E: 500 optimizer steps x 4 accumulated iterations (was 500)
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 24
evaluation = dict(interval=1, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from = None  # Gen2E: was 'ckpts/r101_dcn_fcos3d_pretrain.pth' (see img_backbone.init_cfg)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=1, max_keep_ckpts=3)  # Gen2E: keep the last 3 epochs (disk)
