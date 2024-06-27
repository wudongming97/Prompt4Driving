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
    use_lidar=True, # load LiDAR for debugging, but not for computation
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

# Pay attention to how we change the data augmentation
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
        num_frames_per_sample=3,
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
    motion_prediction=True, # update cross-frame movements
    motion_prediction_ref_update=True, # use forecasting to update cross-frame movements
    num_query=500,
    num_classes=10,
    pc_range=point_cloud_range,
    runtime_tracker=dict(
        output_threshold=0.2,
        score_threshold=0.4,
        record_threshold=0.4,
        max_age_since_update=7,),
    spatial_temporal_reason=dict( # past and future reasoning
        history_reasoning=True, # use past reasoning
        future_reasoning=True,  # use future reasoning
        hist_len=3,
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
    lr=2e-4,
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
load_from='ckpts/f1_q5_e12.pth'
# load_from='ckpts/petr_vovnet_gridmask_p4_800x320.pth'
resume_from=None

# Saving metrics to: /tmp/tmpt91ur17x/results
# mAP: 0.3771
# mATE: 0.6994
# mASE: 0.2722
# mAOE: 0.4251
# mAVE: 0.5899
# mAAE: 0.2170
# NDS: 0.4682
# Eval time: 146.9s

# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.589   0.471   0.151   0.076   0.564   0.215
# truck   0.341   0.709   0.216   0.123   0.514   0.249
# bus     0.411   0.782   0.215   0.091   1.383   0.213
# trailer 0.208   1.025   0.234   0.670   0.291   0.222
# construction_vehicle    0.095   1.024   0.473   1.197   0.157   0.308
# pedestrian      0.446   0.659   0.300   0.514   0.575   0.270
# motorcycle      0.354   0.657   0.246   0.441   0.862   0.234
# bicycle 0.363   0.544   0.256   0.559   0.373   0.023
# traffic_cone    0.511   0.523   0.337   nan     nan     nan
# barrier 0.451   0.599   0.293   0.156   nan     nan
# 2023-03-30 02:42:30,023 - mmdet - INFO - Exp name: f3_q500_800x320.py

# Per-class results:
#                 AMOTA   AMOTP   RECALL  MOTAR   GT      MOTA    MOTP    MT      ML      FAF     TP      FP      FN      IDS     FRAG    TID     LGD
# bicycle         0.328   1.472   0.401   0.766   1993    0.307   0.610   38      80      13.3    799     187     1193    1       6       1.41    1.72
# bus             0.532   1.261   0.643   0.740   2112    0.475   0.829   45      35      22.4    1357    353     755     0       26      1.36    1.95
# car             0.595   1.008   0.655   0.828   58317   0.542   0.556   1661    1257    113.8   38147   6556    20109   61      321     1.06    1.49
# motorcy         0.284   1.440   0.433   0.686   1977    0.295   0.658   27      58      19.2    851     267     1121    5       11      2.01    2.52
# pedestr         0.390   1.420   0.496   0.750   25423   0.370   0.778   457     757     72.2    12543   3135    12817   63      179     1.68    2.39
# trailer         0.154   1.716   0.335   0.525   2425    0.175   1.000   23      88      38.9    808     384     1613    4       20      2.31    3.38
# truck           0.423   1.255   0.586   0.671   9650    0.393   0.742   198     203     49.5    5649    1861    3993    8       72      1.19    1.99

# Aggregated results:
# AMOTA   0.386
# AMOTP   1.367
# RECALL  0.507
# MOTAR   0.709
# GT      14556
# MOTA    0.365
# MOTP    0.739
# MT      2449
# ML      2478
# FAF     47.1
# TP      60154
# FP      12743
# FN      41601
# IDS     142
# FRAG    635
# TID     1.57
# LGD     2.21
# Eval time: 1627.3s

# Rendering curves
# {'label_metrics': {'amota': {'bicycle': 0.3275954315778676, 'bus': 0.5317879556317001, 'car': 0.5954022554620864, 'motorcycle': 0.2835934637763201, 'pedestrian': 0.38962493129398323, 'trailer': 0.15385648041242692, 'truck': 0.42325230766222094}, 'amotp': {'bicycle': 1.472404405791283, 'bus': 1.2606958391954268, 'car': 1.0081252864639592, 'motorcycle': 1.4399019963330724, 'pedestrian': 1.419538567768893, 'trailer': 1.7163245043742077, 'truck': 1.2551209478298095}, 'recall': {'bicycle': 0.4014049172102358, 'bus': 0.6425189393939394, 'car': 0.6551777354802202, 'motorcycle': 0.43297926150733435, 'pedestrian': 0.495850214372812, 'trailer': 0.33484536082474226, 'truck': 0.5862176165803109}, 'motar': {'bicycle': 0.7659574468085106, 'bus': 0.7398673544583639, 'car': 0.8281385167903113, 'motorcycle': 0.6862514688601645, 'pedestrian': 0.7500597943075821, 'trailer': 0.5247524752475248, 'truck': 0.6705611612674809}, 'gt': {'bicycle': 1993.0, 'bus': 2112.0, 'car': 58317.0, 'motorcycle': 1977.0, 'pedestrian': 25423.0, 'trailer': 2425.0, 'truck': 9650.0}, 'mota': {'bicycle': 0.30707476166583036, 'bus': 0.47537878787878785, 'car': 0.5417116792701957, 'motorcycle': 0.2953970662620131, 'pedestrian': 0.3700586083467726, 'trailer': 0.17484536082474222, 'truck': 0.39253886010362693}, 'motp': {'bicycle': 0.6096479851160349, 'bus': 0.8293627798066104, 'car': 0.555636368847498, 'motorcycle': 0.6575105013685492, 'pedestrian': 0.7776470542205001, 'trailer': 0.9996389890988224, 'truck': 0.7418786573755172}, 'mt': {'bicycle': 38.0, 'bus': 45.0, 'car': 1661.0, 'motorcycle': 27.0, 'pedestrian': 457.0, 'trailer': 23.0, 'truck': 198.0}, 'ml': {'bicycle': 80.0, 'bus': 35.0, 'car': 1257.0, 'motorcycle': 58.0, 'pedestrian': 757.0, 'trailer': 88.0, 'truck': 203.0}, 'faf': {'bicycle': 13.28125, 'bus': 22.441195168467896, 'car': 113.81944444444446, 'motorcycle': 19.181034482758623, 'pedestrian': 72.16850828729282, 'trailer': 38.94523326572008, 'truck': 49.547390841320556}, 'tp': {'bicycle': 799.0, 'bus': 1357.0, 'car': 38147.0, 'motorcycle': 851.0, 'pedestrian': 12543.0, 'trailer': 808.0, 'truck': 5649.0}, 'fp': {'bicycle': 187.0, 'bus': 353.0, 'car': 6556.0, 'motorcycle': 267.0, 'pedestrian': 3135.0, 'trailer': 384.0, 'truck': 1861.0}, 'fn': {'bicycle': 1193.0, 'bus': 755.0, 'car': 20109.0, 'motorcycle': 1121.0, 'pedestrian': 12817.0, 'trailer': 1613.0, 'truck': 3993.0}, 'ids': {'bicycle': 1.0, 'bus': 0.0, 'car': 61.0, 'motorcycle': 5.0, 'pedestrian': 63.0, 'trailer': 4.0, 'truck': 8.0}, 'frag': {'bicycle': 6.0, 'bus': 26.0, 'car': 321.0, 'motorcycle': 11.0, 'pedestrian': 179.0, 'trailer': 20.0, 'truck': 72.0}, 'tid': {'bicycle': 1.4102564102564104, 'bus': 1.3625, 'car': 1.0598171701112877, 'motorcycle': 2.013157894736842, 'pedestrian': 1.6804878048780487, 'trailer': 2.31, 'truck': 1.1857142857142857}, 'lgd': {'bicycle': 1.7243589743589745, 'bus': 1.95, 'car': 1.4922496025437202, 'motorcycle': 2.5197368421052633, 'pedestrian': 2.386341463414634, 'trailer': 3.38, 'truck': 1.987142857142857}}, 'eval_time': 1627.2693967819214, 'cfg': {'tracking_names': ['bicycle', 'bus', 'car', 'motorcycle', 'pedestrian', 'trailer', 'truck'], 'pretty_tracking_names': {'bicycle': 'Bicycle', 'bus': 'Bus', 'car': 'Car', 'motorcycle': 'Motorcycle', 'pedestrian': 'Pedestrian', 'trailer': 'Trailer', 'truck': 'Truck'}, 'tracking_colors': {'bicycle': 'C9', 'bus': 'C2', 'car': 'C0', 'motorcycle': 'C6', 'pedestrian': 'C5', 'trailer': 'C3', 'truck': 'C1'}, 'class_range': {'car': 50, 'truck': 50, 'bus': 50, 'trailer': 50, 'pedestrian': 40, 'motorcycle': 40, 'bicycle': 40}, 'dist_fcn': 'center_distance', 'dist_th_tp': 2.0, 'min_recall': 0.1, 'max_boxes_per_sample': 500, 'metric_worst': {'amota': 0.0, 'amotp': 2.0, 'recall': 0.0, 'motar': 0.0, 'mota': 0.0, 'motp': 2.0, 'mt': 0.0, 'ml': -1.0, 'faf': 500, 'gt': -1, 'tp': 0.0, 'fp': -1.0, 'fn': -1.0, 'ids': -1.0, 'frag': -1.0, 'tid': 20, 'lgd': 20}, 'num_thresholds': 40}, 'amota': 0.38644468940237214, 'amotp': 1.3674445068223788, 'recall': 0.5069991493385135, 'motar': 0.7093697453914197, 'gt': 14556.714285714286, 'mota': 0.3652864463359956, 'motp': 0.7387603336905046, 'mt': 2449.0, 'ml': 2478.0, 'faf': 47.05486521285779, 'tp': 60154.0, 'fp': 12743.0, 'fn': 41601.0, 'ids': 142.0, 'frag': 635.0, 'tid': 1.5745619379566962, 'lgd': 2.205689962795064, 'meta': {'use_lidar': True, 'use_camera': True, 'use_radar': False, 'use_map': False, 'use_external': False}}
# {'pts_bbox_NuScenes/amota': 0.38644468940237214, 'pts_bbox_NuScenes/amotp': 1.3674445068223788, 'pts_bbox_NuScenes/recall': 0.5069991493385135, 'pts_bbox_NuScenes/motar': 0.7093697453914197, 'pts_bbox_NuScenes/gt': 14556.714285714286, 'pts_bbox_NuScenes/mota': 0.3652864463359956, 'pts_bbox_NuScenes/motp': 0.7387603336905046, 'pts_bbox_NuScenes/mt': 2449.0, 'pts_bbox_NuScenes/ml': 2478.0, 'pts_bbox_NuScenes/faf': 47.05486521285779, 'pts_bbox_NuScenes/tp': 60154.0, 'pts_bbox_NuScenes/fp': 12743.0, 'pts_bbox_NuScenes/fn': 41601.0, 'pts_bbox_NuScenes/ids': 142.0, 'pts_bbox_NuScenes/frag': 635.0, 'pts_bbox_NuScenes/tid': 1.5745619379566962, 'pts_bbox_NuScenes/lgd': 2.205689962795064}
# liuyingfei@waymobevtools-ps9nm-1906635-worker-0:/data/PF-Track$
