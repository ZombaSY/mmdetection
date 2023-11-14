_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn-hair.py',
    '../_base_/datasets/coco_instance-hair.py',
    '../_base_/schedules/schedule_1x-hair.py', '../_base_/default_runtime-hair.py'
]

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa

model = dict(
    type='MaskRCNN',
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[96, 192, 384, 768]))

# augmentation strategy originates from DETR / Sparse RCNN
# train_pipeline = [
#     dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
#     dict(type='RandomFlip', prob=0.5),
#     dict(
#         type='RandomChoice',
#         transforms=[[
#             dict(
#                 type='RandomChoiceResize',
#                 scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
#                         (608, 1333), (640, 1333), (672, 1333), (704, 1333),
#                         (736, 1333), (768, 1333), (800, 1333)],
#                 keep_ratio=True)
#         ],
#                     [
#                         dict(
#                             type='RandomChoiceResize',
#                             scales=[(400, 1333), (500, 1333), (600, 1333)],
#                             keep_ratio=True),
#                         dict(
#                             type='RandomCrop',
#                             crop_type='absolute_range',
#                             crop_size=(384, 600),
#                             allow_negative_crop=True),
#                         dict(
#                             type='RandomChoiceResize',
#                             scales=[(480, 1333), (512, 1333), (544, 1333),
#                                     (576, 1333), (608, 1333), (640, 1333),
#                                     (672, 1333), (704, 1333), (736, 1333),
#                                     (768, 1333), (800, 1333)],
#                             keep_ratio=True)
#                     ]]),
#     dict(type='PackDetInputs')
# ]
# train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
