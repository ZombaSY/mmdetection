auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
base_lr = 0.001
data_root = '/home/sunyong/workspace/Documents/GitHub/Scalp-python/V2/data/tmp'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=5, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
epochs = 200
launcher = 'pytorch'
load_from = 'pretrained/cascade-mask-rcnn_x101-32x4d-dconv-c3-c5_fpn_1x_coco/epoch_40.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
metainfo = dict(
    classes=(
        'hair1',
        'hair2',
        'hair3',
        'hair4',
        'hair5',
    ),
    palette=[
        (
            0,
            0,
            0,
        ),
        (
            0,
            0,
            255,
        ),
        (
            0,
            255,
            0,
        ),
        (
            0,
            255,
            255,
        ),
        (
            255,
            0,
            0,
        ),
    ])
model = dict(
    backbone=dict(
        base_width=4,
        dcn=dict(deform_groups=1, fallback_on_stride=False, type='DCN'),
        depth=101,
        frozen_stages=1,
        groups=32,
        init_cfg=dict(
            checkpoint='open-mmlab://resnext101_32x4d', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        stage_with_dcn=(
            False,
            True,
            True,
            True,
        ),
        style='pytorch',
        type='ResNeXt'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=True,
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        type='FPN'),
    roi_head=dict(
        bbox_head=[
            dict(
                bbox_coder=dict(
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.1,
                        0.1,
                        0.2,
                        0.2,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(beta=1.0, loss_weight=1.0, type='SmoothL1Loss'),
                loss_cls=dict(
                    loss_weight=1.0,
                    type='CrossEntropyLoss',
                    use_sigmoid=False),
                num_classes=80,
                reg_class_agnostic=True,
                roi_feat_size=7,
                type='Shared2FCBBoxHead'),
            dict(
                bbox_coder=dict(
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.05,
                        0.05,
                        0.1,
                        0.1,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(beta=1.0, loss_weight=1.0, type='SmoothL1Loss'),
                loss_cls=dict(
                    loss_weight=1.0,
                    type='CrossEntropyLoss',
                    use_sigmoid=False),
                num_classes=80,
                reg_class_agnostic=True,
                roi_feat_size=7,
                type='Shared2FCBBoxHead'),
            dict(
                bbox_coder=dict(
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.033,
                        0.033,
                        0.067,
                        0.067,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(beta=1.0, loss_weight=1.0, type='SmoothL1Loss'),
                loss_cls=dict(
                    loss_weight=1.0,
                    type='CrossEntropyLoss',
                    use_sigmoid=False),
                num_classes=80,
                reg_class_agnostic=True,
                roi_feat_size=7,
                type='Shared2FCBBoxHead'),
        ],
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        mask_head=dict(
            conv_out_channels=256,
            in_channels=256,
            loss_mask=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_mask=True),
            num_classes=80,
            num_convs=4,
            type='FCNMaskHead'),
        mask_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=14, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        num_stages=3,
        stage_loss_weights=[
            1,
            0.5,
            0.25,
        ],
        type='CascadeRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales=[
                8,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(
            beta=0.1111111111111111, loss_weight=1.0, type='SmoothL1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        type='RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            mask_thr_binary=0.5,
            max_per_img=100,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.05),
        rpn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=1000)),
    train_cfg=dict(
        rcnn=[
            dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=False,
                    min_pos_iou=0.5,
                    neg_iou_thr=0.5,
                    pos_iou_thr=0.5,
                    type='MaxIoUAssigner'),
                debug=False,
                mask_size=28,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=512,
                    pos_fraction=0.25,
                    type='RandomSampler')),
            dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=False,
                    min_pos_iou=0.6,
                    neg_iou_thr=0.6,
                    pos_iou_thr=0.6,
                    type='MaxIoUAssigner'),
                debug=False,
                mask_size=28,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=512,
                    pos_fraction=0.25,
                    type='RandomSampler')),
            dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=False,
                    min_pos_iou=0.7,
                    neg_iou_thr=0.7,
                    pos_iou_thr=0.7,
                    type='MaxIoUAssigner'),
                debug=False,
                mask_size=28,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=512,
                    pos_fraction=0.25,
                    type='RandomSampler')),
        ],
        rpn=dict(
            allowed_border=0,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='RandomSampler')),
        rpn_proposal=dict(
            max_per_img=2000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=2000)),
    type='CascadeRCNN')
num_last_epochs = 66
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.001, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(decay_mult=1.0, lr_mult=0.1))),
    type='OptimWrapper')
param_scheduler = [
    dict(
        T_max=134,
        begin=5,
        by_epoch=True,
        convert_to_iter_based=True,
        end=134,
        eta_min=5e-05,
        type='CosineAnnealingLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file=
        '/home/sunyong/workspace/Documents/GitHub/Scalp-python/V2/data/tmp/anno/coco_test.json',
        data_prefix=dict(img='input/'),
        data_root=data_root,
        metainfo=dict(
            classes=(
                'hair1',
                'hair2',
                'hair3',
                'hair4',
                'hair5',
            ),
            palette=[
                (
                    0,
                    0,
                    0,
                ),
                (
                    0,
                    0,
                    255,
                ),
                (
                    0,
                    255,
                    0,
                ),
                (
                    0,
                    255,
                    255,
                ),
                (
                    255,
                    0,
                    0,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/home/sunyong/workspace/Documents/GitHub/Scalp-python/V2/data/tmp/anno/coco_test.json',
    format_only=True,
    metric=[
        'bbox',
        'segm',
    ],
    outfile_prefix='./work_dirs/coco_instance/test',
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        640,
        640,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=200, type='EpochBasedTrainLoop', val_interval=5)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=8,
    dataset=dict(
        ann_file='annotations/coco_train_trim.json',
        backend_args=None,
        data_prefix=dict(img='train_image/'),
        data_root=data_root,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        metainfo=dict(
            classes=(
                'hair1',
                'hair2',
                'hair3',
                'hair4',
                'hair5',
            ),
            palette=[
                (
                    0,
                    0,
                    0,
                ),
                (
                    0,
                    0,
                    255,
                ),
                (
                    0,
                    255,
                    0,
                ),
                (
                    0,
                    255,
                    255,
                ),
                (
                    255,
                    0,
                    0,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(
                poly2mask=False,
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True),
            dict(
                img_scale=(
                    640,
                    640,
                ),
                max_cached_images=20,
                pad_val=114.0,
                prob=0.5,
                random_pop=False,
                type='CachedMosaic'),
            dict(
                img_scale=(
                    640,
                    640,
                ),
                max_cached_images=10,
                pad_val=(
                    114,
                    114,
                    114,
                ),
                prob=0.5,
                random_pop=False,
                ratio_range=(
                    1.0,
                    1.0,
                ),
                type='CachedMixUp'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.8,
                    1.2,
                ),
                scale=(
                    768,
                    768,
                ),
                type='RandomResize'),
            dict(crop_size=(
                640,
                640,
            ), type='RandomCrop'),
            dict(type='YOLOXHSVRandomAug'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        poly2mask=False,
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True),
    dict(
        img_scale=(
            640,
            640,
        ),
        max_cached_images=20,
        pad_val=114.0,
        prob=0.5,
        random_pop=False,
        type='CachedMosaic'),
    dict(
        img_scale=(
            640,
            640,
        ),
        max_cached_images=10,
        pad_val=(
            114,
            114,
            114,
        ),
        prob=0.5,
        random_pop=False,
        ratio_range=(
            1.0,
            1.0,
        ),
        type='CachedMixUp'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.8,
            1.2,
        ),
        scale=(
            768,
            768,
        ),
        type='RandomResize'),
    dict(crop_size=(
        640,
        640,
    ), type='RandomCrop'),
    dict(type='YOLOXHSVRandomAug'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='/home/sunyong/workspace/Documents/GitHub/Scalp-python/V2/data/tmp/anno/coco_test.json',
        backend_args=None,
        data_prefix=dict(img='input/'),
        data_root=data_root,
        metainfo=dict(
            classes=(
                'hair1',
                'hair2',
                'hair3',
                'hair4',
                'hair5',
            ),
            palette=[
                (
                    0,
                    0,
                    0,
                ),
                (
                    0,
                    0,
                    255,
                ),
                (
                    0,
                    255,
                    0,
                ),
                (
                    0,
                    255,
                    255,
                ),
                (
                    255,
                    0,
                    0,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    '/home/sunyong/workspace/Documents/GitHub/Scalp-python/V2/data/tmp/anno/coco_test.json',
    backend_args=None,
    format_only=False,
    metric=[
        'bbox',
        'segm',
    ],
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/cascade-mask-rcnn_x101-32x4d-dconv-c3-c5_fpn_1x_coco'
