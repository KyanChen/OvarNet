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

# attribute_index_file = dict(
#     file=data_root+'/VAW/common2common_att2id.json',
#     att_group='common2'
# )

attribute_index_file = dict(
    att_file='../attributes/VAW/common2common_att2id.json',
    att_group='common1+common2',
    # att_file='../attributes/VAW/common2rare_att2id.json',
    # att_group='common+rare',
    # att_file='../attributes/OVAD/common2common_att2id.json',
    # att_group='common1',
    # category_file='../attributes/COCO/common2common_category2id_48_17.json',
    # # category_file='../attributes/COCO/common2common_category2id_48_32.json',
    # category_group='common1+common2',
)

# attribute_index_file = dict(
#     file=data_root+'/VAW/common2rare_att2id.json',
#     att_group='all'
# )
model = dict(
    type='CLIP_Tester',
    attribute_index_file=attribute_index_file,
    backbone_name='RN50',  # RN101, RN50x4，RN50x64, ViT-B/16, ViT-L/14@336px, ViT-B/16
)

img_scale = (224, 224)  # (224, 224) (288, 288) (336, 336), (384, 384) (448, 448)
# dataset settings
dataset_type = 'VAWCropDataset'
img_norm_cfg = dict(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711],
    to_rgb=False
)
# Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, rearrange=True, channel_order='rgb'),
    dict(type='ScaleCrop', scale_range=[0.0, 0.3]),
    dict(type='RandomCrop', crop_size=[0.8, 0.8], crop_type='relative_range'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=img_scale, center_pad=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_labels']),
    dict(type='Collect', keys=['img', 'gt_labels', 'data_set_type'])
]

train_generated_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, rearrange=True, channel_order='rgb'),
    dict(type='RandomCrop', crop_size=[0.7, 0.7], crop_type='relative_range'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=img_scale, center_pad=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_labels']),
    dict(type='Collect', keys=['img', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, rearrange=True, channel_order='rgb'),
    dict(type='ScaleCrop', scale_range=[0.0, 0.2]),
    dict(type='RandomCrop', crop_size=[0.9, 0.9], crop_type='relative_range'),
    dict(type='MultiScaleFlipAug',
         img_scale=img_scale,
         flip=False,
         transforms=[
            dict(type='Resize', img_scale=img_scale, keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=img_scale, center_pad=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]
    )
]

test_generated_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, rearrange=True, channel_order='rgb'),
    dict(type='MultiScaleFlipAug',
         img_scale=img_scale,
         flip=False,
         transforms=[
            dict(type='Resize', img_scale=img_scale, keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=img_scale, center_pad=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]
    )
]

samples_per_gpu = 512
data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=8,
    persistent_workers=True,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        dataset_split='train',
        attribute_index_file=attribute_index_file,
        # attribute_index_file=dict(
        #     file=data_root+'/VAW/common2rare_att2id.json',
        #     att_group='rare'
        # ),
        dataset_names=['coco', 'vaw'],
        save_label=False,
        load_label=None,
        test_mode=False,
        open_category=False,
        pipeline=train_pipeline
        # pipeline=train_generated_pipeline
    ),
    val=dict(
        samples_per_gpu=samples_per_gpu,
        type=dataset_type,
        data_root=data_root,
        dataset_split='test',
        attribute_index_file=attribute_index_file,
        dataset_names=['coco', 'vaw'],
        test_mode=True,
        open_category=False,
        pipeline=test_pipeline),
    test=dict(
        samples_per_gpu=256,
        type=dataset_type,
        data_root=data_root,
        dataset_split='test',
        attribute_index_file=dict(
            # att_file='../attributes/VAW/common2rare_att2id.json',
            # att_group='common+rare',
            att_file='../attributes/VAW/common2common_att2id.json',
            att_group='common1+common2',
            # att_file='../attributes/OVAD/common2common_att2id.json',
            # att_group='common1',
            # category_file='../attributes/COCO/common2common_category2id_48_17.json',
            # # # # category_file='../attributes/COCO/common2common_category2id_48_32.json',
            # category_group='common1+common2',
        ),
        test_mode=True,
        open_category=False,
        dataset_names=['vaw'],
        save_label=False,
        load_label=None,
        pipeline=test_pipeline
        # pipeline=test_generated_pipeline
    )
)

# optimizer
optimizer = dict(
    constructor='SubModelConstructor',
    sub_model={
        'prompt_att_learner': {},
        # 'prompt_category_learner': {},
        # 'image_encoder': {'lr_mult': 0.1},
        'text_encoder': {'lr_mult': 0.1},
        'bbox_head': {}, 'logit_scale': {}
    },
    # type='SGD',
    # lr=1e-4,
    # # momentum=0.9,
    # weight_decay=0.0005,
    type='AdamW',
    lr=1e-4,
    weight_decay=0.0005
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

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.1,
    # gamma=0.5,
    # step=[50, 80],
    step=[35, 50]
)
# lr_config = None
# lr_config = dict(
#     policy='CosineAnnealing',
#     by_epoch=False,
#     min_lr_ratio=1e-2,
#     warmup='linear',
#     warmup_ratio=1e-3,
#     warmup_iters=1,
#     warmup_by_epoch=True)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=60)
evaluation = dict(interval=5, metric='mAP')

# load_from = 'results/EXP20220903_0/epoch_20.pth'
load_from = None
resume_from = None