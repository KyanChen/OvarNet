checkpoint_config = dict(interval=15)
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
# data_root = 'D:/Dataset'
data_root = '/data/kyanchen/prompt/data'
model = dict(
    type='CLIP_Prompter',
    classname_path=data_root+'/VAW/attribute_index.json',
    backbone=dict(
        type='CLIPModel',
        backbone_name='RN50',
        # backbone_name='ViT-B/16',
        load_ckpt_from=None,
        precision='fp16',
    ),
    prompt_learner=dict(
        type='PromptLearner',
        n_ctx=16,
        ctx_init='',
        c_specific=False,
        class_token_position='middle'
    ),
    neck=None,
    bbox_head=dict(
        type='PromptHead',
        data_root=data_root,
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            ignore_index=2,
            avg_non_ignore=True,
            loss_weight=1.0,
            reduction='mean'
        )
    )
)
# dataset settings
dataset_type = 'VAWProposalDataset'
img_norm_cfg = dict(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711],
    to_rgb=False
)
# Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, rearrange=True, channel_order='rgb'),
    dict(type='ScaleCrop', scale_range=[0.2, 0.4]),
    dict(type='RandomCrop', crop_size=[0.7, 0.7], crop_type='relative_range'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=(224, 224), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(224, 224), center_pad=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_labels']),
    dict(type='Collect', keys=['img', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, rearrange=True, channel_order='rgb'),
    dict(type='ScaleCrop', scale_range=[0.2, 0.4]),
    dict(type='RandomCrop', crop_size=[0.7, 0.7], crop_type='relative_range'),
    dict(type='MultiScaleFlipAug',
         img_scale=(224, 224),
         flip=False,
         transforms=[
            dict(type='Resize', img_scale=(224, 224), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(224, 224)),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]
    )
]

num_shots = 'all'
seed = 1
data = dict(
    samples_per_gpu=256,
    workers_per_gpu=0,
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
    test=dict(
        samples_per_gpu=256,
        type=dataset_type,
        data_root=data_root,
        pattern='test',
        test_mode=True,
        pipeline=test_pipeline
    )
)
#
# optimizer
optimizer = dict(
    constructor='SubModelConstructor',
    sub_model='prompt_learner',
    type='SGD',
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# optimizer
# optimizer = dict(
#     constructor='SubModelConstructor',
#     sub_model='prompt_learner',
#     type='AdamW',
#     lr=1e-4,
#     weight_decay=1e-3
# )
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# # # learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=2000,  # same as burn-in in darknet
#     warmup_ratio=0.1,
#     step=[90, 120])

lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=1e-2,
    warmup='linear',
    warmup_ratio=1e-3,
    warmup_iters=1,
    warmup_by_epoch=True)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=150)
evaluation = dict(interval=1000, metric='mAP')

load_from = None
resume_from = None