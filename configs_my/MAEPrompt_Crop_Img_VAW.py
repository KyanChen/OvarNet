checkpoint_config = dict(interval=20)
# yapf:disable
log_config = dict(
    interval=30,
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
find_unused_parameters = True

# model settings
# data_root = 'D:/Dataset'
data_root = '/data/kyanchen/prompt/data'
model = dict(
    type='CLIP_Prompter',
    classname_path=data_root+'/VAW/attribute_index.json',
    need_train_names=['prompt_learner', 'img_proj_head',
                      'image_encoder',
                      # 'text_encoder',
                      'bbox_head', 'logit_scale'],
    # need_train_names=['prompt_learner', 'neck', 'roi_head', 'bbox_head', 'logit_scale'],
    img_encoder=dict(
        type='VisionTransformer',
        arch='vit_base_patch16',
        load_pretrain='../pretrain/dino_vitbase16_pretrain.pth'
        # load_pretrain='../pretrain/mae_pretrain_vit_base.pth',
    ),
    backbone=dict(
        type='CLIPModel',
        backbone_name='RN50',
        with_attn=True,
        # backbone_name='ViT-B/16',
        load_ckpt_from=None,
        precision='fp32',
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
        re_weight_alpha=0.25,
        re_weight_gamma=2,
        re_weight_beta=0.995,
        balance_unk=0.15
    )
)
# dataset settings
dataset_type = 'VAWCropDataset'
# img_norm_cfg = dict(
#     mean=[0.48145466, 0.4578275, 0.40821073],
#     std=[0.26862954, 0.26130258, 0.27577711],
#     to_rgb=False
# )
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    to_rgb=False
)

# Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, rearrange=True, channel_order='rgb'),
    dict(type='ScaleCrop', scale_range=[0.0, 0.3]),
    dict(type='RandomCrop', crop_size=[0.8, 0.8], crop_type='relative_range'),
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
    dict(type='ScaleCrop', scale_range=[0.0, 0.2]),
    dict(type='RandomCrop', crop_size=[0.9, 0.9], crop_type='relative_range'),
    dict(type='MultiScaleFlipAug',
         img_scale=(224, 224),
         flip=False,
         transforms=[
            dict(type='Resize', img_scale=(224, 224), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(224, 224), center_pad=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]
    )
]


data = dict(
    samples_per_gpu=100,
    workers_per_gpu=8,
    persistent_workers=True,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        pattern='train',
        test_mode=False,
        open_category=False,
        pipeline=train_pipeline),
    val=dict(
        samples_per_gpu=100,
        type=dataset_type,
        data_root=data_root,
        pattern='test',
        test_mode=True,
        open_category=False,
        pipeline=test_pipeline),
    test=dict(
        samples_per_gpu=100,
        type=dataset_type,
        data_root=data_root,
        pattern='test',
        test_mode=True,
        open_category=False,
        pipeline=test_pipeline
    )
)

# optimizer
optimizer = dict(
    constructor='SubModelConstructor',
    # sub_model='prompt_learner',
    # need_train_names = ['prompt_learner', 'text_encoder', 'bbox_head', 'logit_scale']
    # sub_model={'prompt_learner': {}, 'image_encoder': {'lr_mult': 0.1}},
    sub_model={
        'prompt_learner': {},
        'image_encoder': {'lr_mult': 0.1},
        # 'text_encoder': {'lr_mult': 0.1},
        'bbox_head': {}, 'logit_scale': {}, 'img_proj_head': {}
    },
    # type='SGD',
    # lr=0.01,
    # momentum=0.9,
    # weight_decay=0.0005,
    type='AdamW',
    lr=1e-4,
    weight_decay=1e-3

)
#
# # optimizer
# optimizer = dict(
#     constructor='SubModelConstructor',
#     # sub_model='prompt_learner',
#     sub_model={'prompt_learner': {}, 'image_encoder': {'lr_mult': 0.1}},
#     type='AdamW',
#     lr=1e-4,
#     weight_decay=1e-3
# )

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# # learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.1,
    # gamma=0.5,
    step=[60, 75, 90]
)

# lr_config = dict(
#     policy='CosineAnnealing',
#     by_epoch=False,
#     min_lr_ratio=1e-2,
#     warmup='linear',
#     warmup_ratio=1e-3,
#     warmup_iters=1,
#     warmup_by_epoch=True)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
evaluation = dict(interval=10, metric='mAP')

load_from = None
resume_from = None