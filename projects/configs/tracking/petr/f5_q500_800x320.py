_base_ = [
    '../../runtime.py'
]


workflow = [('train', 1)]
plugin = True
plugin_dir = 'projects/'

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)

class_names = [
    'car', 'truck', 'bus', 'trailer', 
    'motorcycle', 'bicycle', 'pedestrian', 
    'construction_vehicle', 'traffic_cone', 'barrier'
]

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)


dataset_type = 'NuScenesTrackingDataset'
data_root = './data/nuscenes/'

file_client_args = dict(backend='disk')
ida_aug_conf = {
        "resize_lim": (0.47, 0.625),
        "final_dim": (320, 800),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": False,
    }

train_pipeline = [
    dict(
        type='LoadPointsFromFile', # visualization purpose
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    # dict(
    #     type='LoadPointsFromMultiSweeps',
    #     sweeps_num=3,
    #     file_client_args=file_client_args),
    dict(type='LoadMultiViewImageFromFiles'),
    dict(type='TrackLoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_forecasting=True),
    dict(type='TrackInstanceRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='TrackObjectNameFilter', classes=class_names),
]

train_pipeline_multiframe = [
    dict(type='TrackResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=True),
    # dict(type='TrackGlobalRotScaleTransImage',
    #         rot_range=[-0.3925, 0.3925],
    #         translation_std=[0, 0, 0],
    #         scale_ratio_range=[0.95, 1.05],
    #         reverse_angle=True,
    #         training=True
    #         ),
    dict(type='TrackNormalizeMultiviewImage', **img_norm_cfg),
    dict(type='TrackPadMultiViewImage', size_divisor=32),
    dict(type='FormatBundle3DTrack'),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'instance_inds', 'img', 'timestamp', 'l2g_r_mat', 'l2g_t', 'l2g',
                                  'gt_forecasting_locs', 'gt_forecasting_masks', 'gt_forecasting_types'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles'),
    # dict(
    #     type='MultiScaleFlipAug3D',
    #     img_scale=(1333, 800),
    #     pts_scale_ratio=1,
    #     flip=False,
    #     transforms=[
    #         dict(
    #             type='DefaultFormatBundle3D',
    #             class_names=class_names,
    #             with_label=False),
    #         dict(type='Collect3D', keys=['img'])
    #     ])
]

test_pipeline_multiframe = [
    dict(type='TrackResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=False),
    dict(type='TrackNormalizeMultiviewImage', **img_norm_cfg),
    dict(type='TrackPadMultiViewImage', size_divisor=32),
    dict(type='FormatBundle3DTrack'),
    dict(type='Collect3D', keys=['img', 'timestamp', 'l2g_r_mat', 'l2g_t', 'l2g'])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        num_frames_per_sample=5,
        forecasting=True,
        data_root=data_root,
        ann_file=data_root + 'tracking_forecasting_infos_train.pkl',
        pipeline=train_pipeline,
        pipeline_multiframe=train_pipeline_multiframe,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR'),
    # ),
    val=dict(type=dataset_type, pipeline=test_pipeline, 
             pipeline_multiframe=test_pipeline_multiframe,
             data_root=data_root, test_mode=True,
             classes=class_names, modality=input_modality,
             ann_file=data_root + 'tracking_forecasting_infos_val.pkl',
             num_frames_per_sample=3,), # when inference, set bs=1
    test=dict(type=dataset_type, pipeline=test_pipeline, 
             pipeline_multiframe=test_pipeline_multiframe,
             data_root=data_root, test_mode=True,
             classes=class_names, modality=input_modality,
             ann_file=data_root + 'tracking_forecasting_infos_val.pkl',
             num_frames_per_sample=3,), # when inference, set bs=1 
    test_tracking=dict(type=dataset_type, pipeline=test_pipeline,
              pipeline_multiframe=test_pipeline_multiframe,
              data_root=data_root, test_mode=True,
              classes=class_names, modality=input_modality,
              ann_file=data_root + 'tracking_forecasting_infos_val.pkl',
              num_frames_per_sample=1,),
    visualization=dict(type=dataset_type, pipeline=train_pipeline,
              pipeline_multiframe=train_pipeline_multiframe,
              data_root=data_root, test_mode=False,
              classes=class_names, modality=input_modality,
              ann_file=data_root + 'tracking_forecasting-mini_infos_val.pkl',
              num_frames_per_sample=1,))


model = dict(
    type='Cam3DTracker',
    tracking=True,
    train_backbone=True,
    use_grid_mask=True,
    if_update_ego=True, # update the ego-motion
    motion_prediction=True,
    motion_prediction_ref_update=True,
    num_query=500,
    num_classes=10,
    pc_range=point_cloud_range,
    runtime_tracker=dict(
        output_threshold=0.2,
        score_threshold=0.4,
        record_threshold=0.4,
        max_age_since_update=7,),
    spatial_temporal_reason=dict(
        history_reasoning=True,
        future_reasoning=True,
        hist_len=5,
        fut_len=8,
        pc_range=point_cloud_range,
        hist_temporal_transformer=dict(
            type='TemporalTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=2,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    with_cp=False,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        spatial_transformer=dict(
            type='TemporalTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=2,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    with_cp=False,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        fut_temporal_transformer=dict(
            type='TemporalTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=2,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    with_cp=False,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),),
    img_backbone=dict(
        type='VoVNetCP',
        spec_name='V-99-eSE',
        norm_eval=True,
        frozen_stages=-1,
        input_ch=3,
        out_features=('stage4','stage5',)),
    img_neck=dict(
        type='CPFPN',
        in_channels=[768, 1024],
        out_channels=256,
        num_outs=2),
    pts_bbox_head=dict(
        type='PETRCamTrackingHead',
        num_classes=10,
        in_channels=256,
        LID=True,
        with_position=True,
        with_multiview=True,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        normedlinear=False,
        transformer=dict(
            type='PETRTrackingTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    with_cp=False,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        bbox_coder=dict(
            type='TrackNMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=150,
            voxel_size=voxel_size,
            score_threshold=0.0,
            num_classes=10), 
        positional_encoding=dict(
            type='SinePositionalEncoding3D', num_feats=128, normalize=True),
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
            pc_range=point_cloud_range))),
    loss=dict(
        type='TrackingLossCombo',
        num_classes=10,
        interm_loss=True,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_prediction=dict(type='L1Loss', loss_weight=0.5),
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
            pc_range=point_cloud_range)))

optimizer = dict(
    type='AdamW', 
    lr=1e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512., grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    )


total_epochs = 12
evaluation = dict(interval=12, pipeline=test_pipeline)
find_unused_parameters=False

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from='ckpts/f3_q5_e12.pth'
resume_from=None