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

attribute_index_file = dict(
    # att_file='../attributes/VAW/common2common_att2id.json',
    # att_group='common1+common2',
    att_file='../attributes/VAW/base2novel_att2id.json',
    att_group='base+novel',
    # att_file='../attributes/VAW/common2rare_att2id.json',
    # att_group='common+rare',
    # att_file='../attributes/OVAD/common2common_att2id.json',
    # att_group='common1',
    # category_file='../attributes/COCO/common2common_category2id_48_17.json',
    # # category_file='../attributes/COCO/common2common_category2id_48_32.json',
    # category_group='common1+common2',
    category_file='../attributes/COCO/common2common_category2id_48_17.json',
    # category_file='../attributes/COCO/common2common_category2id_48_32.json',
    category_group='common1+common2',
)


model = dict(
    type='CLIPAttr_Booster',
    attribute_index_file=attribute_index_file,
    gather_gpus=True,  # train_text:32, train_all:12
    need_train_names=[
        'prompt_category_learner',
        'prompt_att_learner',
        'image_encoder',
        'text_encoder',
        'prompt_phase_learner',
        'bbox_head', 'logit_scale'
    ],
    backbone=dict(
        type='CLIPModel',
        # backbone_name='RN50',  # RN101, RN50x4，RN50x64, ViT-B/16, ViT-L/14@336px, ViT-B/16
        with_attn=True,
        backbone_name='ViT-B/16',
        load_ckpt_from=None,
        precision='fp32',
    ),
    shared_prompt_vectors=True,
    prompt_att_learner=dict(
        type='PromptAttributes',
        # load_ckpt_from='results/EXP20221006_0/epoch_20.pth',
        prompt_config=dict(
            n_prompt=30,
            is_att_specific=False,
            att_position='mid',
            att2type='../attributes/VAW/att2types.json',
            # att2type=None,
            # att2type='../attributes/OVAD/att2types.json',
            context_length=77,
            n_prompt_type=None,
            generated_context=False,
            pos_emb=False,
        ),
    ),
    prompt_category_learner=dict(
        type='PromptAttributes',
        # load_ckpt_from='results/EXP20221006_0/epoch_20.pth',
        prompt_config=dict(
            n_prompt=30,
            is_att_specific=False,
            att_position='mid',
            att2type='../attributes/COCO/category2types.json',
            # att2type=None,
            context_length=77,
            n_prompt_type=None,
            generated_context=False,
            pos_emb=False,
        ),
    ),
    prompt_phase_learner=dict(
        type='PromptPhases',
        prompt_config=dict(
            n_prompt=16,
            att_position='mid',
            context_length=77,
        ),
    ),
    prompt_caption_learner=dict(
        type='PromptCaption',
        prompt_config=dict(
            context_length=77,
        ),
    ),
    neck=None,
    mil_loss=dict(
        type='MILCrossEntropy'
    ),
    bbox_head=dict(
        type='PromptHead',
        attr_freq_file='../attributes/VAW/attr_freq_wo_sort.json',
        category_freq_file='../attributes/COCO/category_freq_wo_sort.json',
        re_weight_different_att=0.25,
        re_weight_category=1,  # 2太大了，出现cate增，att下降
        re_weight_gamma=2,
        re_weight_beta=0.995,
        balance_unk=0.25,  # boost: 0.5; Cap,VAW,COCO: 0.2
        balance_capdata=0.5,
        # balance_unk=1  # gen
    )
)

img_scale = (224, 224)  # (224, 224) (288, 288) (336, 336), (384, 384) (448, 448)
# dataset settings
dataset_type = 'BoostCLIPWithCapCropDataset'
img_norm_cfg = dict(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711],
    to_rgb=False
)

train_vawcoco_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, rearrange=True, channel_order='rgb'),
    dict(type='ScaleCrop', scale_range=[0.0, 0.3]),
    dict(type='RandomCrop', crop_size=[0.8, 0.8], crop_type='relative_range'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=img_scale, center_pad=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_labels']),
    dict(type='Collect', keys=['img', 'gt_labels', 'data_set_type', 'phase', 'caption'])
]

train_cap_collectall_pipeline = [
    dict(type='Collect',
         meta_keys=('filename',
                    'ori_filename',
                    'ori_shape',
                    'img_shape'),
         keys=[
            'img',
            'img_crops', 'crops_logits',
            'crops_labels', 'caption',
             'phases', 'data_set_type', 'gt_labels'
    ])
]

train_cap_imgcrops_pipeline = [
    dict(type='ScaleCrop', scale_range=[0.0, 0.1]),
    dict(type='RandomCrop', crop_size=[0.9, 0.9], crop_type='relative_range'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=img_scale, center_pad=True),
    dict(type='ImageToTensor', keys=['img']),
    # dict(type='ToTensor', keys=['gt_labels']),
    dict(type='Collect', keys=['img'])
]

train_cap_biggestproposal_pipeline = [
    dict(type='ScaleCrop', scale_range=[0.0, 0.01]),
    # dict(type='RandomCrop', crop_size=[0.8, 0.8], crop_type='relative_range'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=img_scale, center_pad=True),
    dict(type='ImageToTensor', keys=['img']),
    # dict(type='ToTensor', keys=['gt_labels']),
    dict(type='Collect', keys=['img'])
]

train_cap_wholeimg_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, rearrange=True, channel_order='rgb'),
    # dict(type='ScaleCrop', scale_range=[0.0, 0.0]),
    # dict(type='RandomCrop', crop_size=[0.8, 0.8], crop_type='relative_range'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=img_scale, center_pad=True),
    dict(type='ImageToTensor', keys=['img']),
    # dict(type='ToTensor', keys=['gt_labels']),
    dict(type='Collect', keys=['img'])
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
# captext:24 capimg:48 coco_captext:80 img:128
samples_per_gpu = 4
data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=4,
    persistent_workers=True,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        dataset_split='train',
        attribute_index_file=attribute_index_file,
        dataset_names=['cococap'],
        # dataset_names=['cc3m'],
        save_label=False,
        load_label=None,
        test_mode=False,
        open_category=False,
        cap_pipeline=[train_cap_wholeimg_pipeline, train_cap_biggestproposal_pipeline, train_cap_imgcrops_pipeline, train_cap_collectall_pipeline],
        vawcoco_pipline=None,
        select_novel=False
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
            att_file='../attributes/VAW/common2rare_att2id.json',
            att_group='common+rare',
            # att_file='../attributes/VAW/common2common_att2id.json',
            # att_group='common1+common2',
            # att_file='../attributes/OVAD/common2common_att2id.json',
            # att_group='common1',
            category_file='../attributes/COCO/common2common_category2id_48_17.json',
            # # # category_file='../attributes/COCO/common2common_category2id_48_32.json',
            category_group='common1+common2',
        ),
        test_mode=True,
        open_category=False,
        dataset_names=['coco', 'vaw'],
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
        # 'prompt_category_learner': {'lr_mult': 0.1},
        'prompt_phase_learner': {},
        'image_encoder': {'lr_mult': 0.1},
        'text_encoder': {'lr_mult': 0.1},
        'bbox_head': {}, 'logit_scale': {}
    },
    # type='SGD',
    # lr=5e-3,
    # momentum=0.9,
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
    step=[50],
    # step=[15, 30]
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
runner = dict(type='EpochBasedRunner', max_epochs=80)
evaluation = dict(interval=5, metric='mAP')

# load_from = 'results/EXP20221006_0/epoch_20.pth'
load_from = 'results/EXP20221109_0/epoch_50.pth'
resume_from = None