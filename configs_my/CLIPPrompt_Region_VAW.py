checkpoint_config = dict(interval=5)
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

# model settings
# data_root = 'D:/Dataset'
data_root = '/data/kyanchen/prompt/data'
model = dict(
    type='CLIP_Prompter_Region',
    classname_path=data_root+'/VAW/attribute_index.json',
    backbone=dict(
        type='CLIPModel',
        backbone_name='RN50',
        with_attn=False,
        out_indices=[1, 2, 3, 4],
        # backbone_name='ViT-B/16',
        load_ckpt_from=None,
        precision='fp32',
    ),
    neck=dict(
        type='FPN',
        # type='RefineChannel',
        in_channels=[256, 512, 1024, 2048],
        # in_channels=[2048],
        out_channels=256,
        num_outs=4),
    roi_head=dict(
        type='ProposalEncoder',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
            # featmap_strides=[32]
            # out_channels=1024,
            # featmap_strides=[32]
        ),
        # shared_head=dict(
        #     type='ResLayer',
        #     depth=50,
        #     stage=3,
        #     stride=1,
        #     norm_eval=False,
        #     inplanes=256,
        #     planes=128,
        # ),
        # in_channels=512,
        in_channels=256,
        out_channels=1024,
    ),
    prompt_learner=dict(
        type='PromptLearner',
        n_ctx=16,
        ctx_init='',
        c_specific=False,
        class_token_position='middle'
    ),
    bbox_head=dict(
        type='PromptHead',
        data_root=data_root,
        re_weight_alpha=0.2,
        re_weight_gamma=2,
        re_weight_beta=0.995,
        balance_unk=0.1
    )
)
# dataset settings
dataset_type = 'VAWRegionDataset'
img_norm_cfg = dict(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711],
    to_rgb=False
)
# Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
# img_size = (512, 512)
img_size = (896, 896)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, rearrange=True, channel_order='rgb'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=img_size, keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=img_size),
    dict(type='Pad', size=img_size, center_pad=True),
    dict(type='RandomExpandAndCropBox', expand_range=(0.95, 1.2), crop_range=(0.85, 1), prob=0.6),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['proposals', 'gt_labels']),
    dict(type='Collect', keys=['img', 'proposals', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, rearrange=True, channel_order='rgb'),
    dict(type='MultiScaleFlipAug',
         img_scale=img_size,
         flip=False,
         transforms=[
            dict(type='Resize', img_scale=img_size, keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='Pad', size=img_size),
            dict(type='Pad', size=img_size,  center_pad=True),
            dict(type='RandomExpandAndCropBox', expand_range=(0.95, 1.2), crop_range=(0.9, 1)),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'proposals'])
        ]
    )
]


data = dict(
    samples_per_gpu=42,
    workers_per_gpu=4,
    # workers_per_gpu=0,
    persistent_workers=True,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        pattern='train',
        test_mode=False,
        pipeline=train_pipeline),
    val=dict(
        samples_per_gpu=42,
        type=dataset_type,
        data_root=data_root,
        pattern='test',
        test_mode=True,
        pipeline=test_pipeline),
    test=dict(
        samples_per_gpu=42,
        type=dataset_type,
        data_root=data_root,
        pattern='test',
        test_mode=True,
        pipeline=test_pipeline
    )
)
# #
# optimizer
optimizer = dict(
    constructor='SubModelConstructor',
    sub_model=['prompt_learner', 'neck', 'roi_head', 'bbox_head'],
    # sub_model={'prompt_learner': {}, 'neck': {}, 'roi_head': {}, 'bbox_head': {}, 'image_encoder': {'lr_mult': 0.01}},
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0005
)

# # optimizer
# optimizer = dict(
#     constructor='SubModelConstructor',
#     sub_model=['prompt_learner', 'neck', 'roi_head'],
#     type='Adam',
#     lr=1e-5,
#     weight_decay=1e-3
# )
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# # learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.1,
    step=[60, 80])

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
# resume_from = 'results/EXP20220707_1/latest.pth'
resume_from = None