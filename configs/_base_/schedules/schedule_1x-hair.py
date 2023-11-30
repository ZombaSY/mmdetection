# training schedule for 1x
epochs = 200
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=epochs, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate

num_last_epochs = epochs // 3
base_lr = 1e-3

# learning rate
param_scheduler = [
    dict(
        # use cosine lr from 5 to 285 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=5,
        T_max=epochs - num_last_epochs,
        end=epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW',
                   lr=base_lr,
                   weight_decay=0.0001,
                   betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))

# optim_wrapper = dict(
#     type='AmpOptimWrapper',
#     constructor='LearningRateDecayOptimizerConstructor',
#     paramwise_cfg={
#         'decay_rate': 0.7,
#         'decay_type': 'layer_wise',
#         'num_layers': 6
#     },
#     optimizer=dict(
#         _delete_=True,
#         type='AdamW',
#         lr=0.0002,
#         betas=(0.9, 0.999),
#         weight_decay=0.05))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
