checkpoint_config = dict(interval=20)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = None
# custom_hooks = [dict(type='SetSubModelEvalHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)

# model settings
img_size = 256
# data_root = 'D:/Dataset'
data_root = '/data/kyanchen/prompt/data'
model = dict(
    type='OFA_Prompter',
    classname_path=data_root+'/VAW/attribute_index.json',
    # ofa_pretrained_weights=data_root+'/../pretrain/ofa_medium.pt',  # 256
    # ofa_pretrained_weights=data_root+'/../pretrain/ofa_tiny.pt',  # 256
    # ofa_pretrained_weights=data_root+'/../pretrain/ofa_base.pt',  # 384
    ofa_pretrained_weights=data_root+'/../pretrain/vqa_base_best.pt',  # 480
    n_sample_attr=4,
    backbone=dict(
        type='OFA',
        ofa_name='ofa_base'
    ),
    prompt_learner=dict(
        type='OFAPromptLearner',
        n_ctx=16,
        ctx_init='',
        c_specific=False,
        class_token_position='end'
    ),
    neck=None,
    bbox_head=dict(
        type='OFAPromptHead',
        data_root=data_root,
        loss_cls=dict(
            type='AdjustLabelSmoothedCrossEntropyCriterion',
            sentence_avg=False,
            label_smoothing=0.1,
            report_accuracy=True
        )
    )
)
# dataset settings
dataset_type = 'VAWDataset'
img_norm_cfg = dict(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5],
    to_rgb=False
)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, rearrange=True, channel_order='rgb'),
    dict(type='ScaleCrop', scale_range=[0.2, 0.4]),
    dict(type='RandomCrop', crop_size=[0.7, 0.7], crop_type='relative_range'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=(img_size, img_size), keep_ratio=True, interpolation='bicubic'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(img_size, img_size), center_pad=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_labels']),
    dict(type='Collect', keys=['img', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, rearrange=True, channel_order='rgb'),
    dict(type='ScaleCrop', scale_range=[0.2, 0.4]),
    dict(type='RandomCrop', crop_size=[0.7, 0.7], crop_type='relative_range'),
    dict(type='MultiScaleFlipAug',
         img_scale=(img_size, img_size),
         flip=False,
         transforms=[
            dict(type='Resize', img_scale=(img_size, img_size), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(img_size, img_size)),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]
    )
]

num_shots = 128
seed = 1
data = dict(
    samples_per_gpu=12,
    workers_per_gpu=2,
    persistent_workers=True,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        num_shots=num_shots,
        pattern='train',
        seed=seed,
        test_mode=False,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        num_shots=num_shots,
        pattern='val',
        seed=seed,
        test_mode=True,
        pipeline=test_pipeline),
    test=
    dict(
        samples_per_gpu=1,
        type=dataset_type,
        data_root=data_root,
        num_shots=num_shots,
        pattern='test',
        seed=seed,
        test_mode=True,
        pipeline=test_pipeline
    )
)

# optimizer
optimizer = dict(
    constructor='SubModelConstructor',
    sub_model='prompt_learner',
    type='SGD',
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0005
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# # learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=2000,  # same as burn-in in darknet
#     warmup_ratio=0.1,
#     step=[218, 246])
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=1e-2,
    warmup='linear',
    warmup_ratio=1e-3,
    warmup_iters=1,
    warmup_by_epoch=True)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
evaluation = dict(interval=1000, metric='mAP', is_logit=False)

load_from = None
# resume_from = 'results/EXP20220523_3/latest.pth'
resume_from = None