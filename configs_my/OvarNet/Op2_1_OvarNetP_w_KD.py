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

# data_root = '/data/kyanchen/prompt/data'
data_root = '/expand_data/datasets'

attribute_index_file = dict(
    # att_file='../attributes/VAW/common2common_att2id.json',
    # att_group='common1+common2',
    att_file='../attributes/VAW/base2novel_att2id.json',
    att_group='base+novel',
    # att_file='../attributes/VAW/base2novel_att2id.json',
    # att_group='base',
    # att_file='../attributes/VAW/common2rare_att2id.json',
    # att_group='common+rare',
    # att_file='../attributes/OVAD/common2common_att2id.json',
    # att_group='common1',
    category_file='../attributes/COCO/common2common_category2id_48_17.json',
    # category_file='../attributes/COCO/common2common_category2id_48_32.json',
    category_group='common1+common2',
    # category_file='../attributes/COCO/common2common_category2id_48_17.json',
    # category_file='../attributes/COCO/common2common_category2id_48_32.json',
    # category_group='common1+common2',
)


# pre_load_ckpt = 'results/EXP20221023_4/epoch_6.pth'  # R50
# pre_load_ckpt = 'results/EXP20221102_0/epoch_6.pth'  # ViT
# pre_load_ckpt = 'results/EXP20221103_1/epoch_6.pth'  # R50
pre_load_ckpt = 'results/EXP20221218_0/epoch_6.pth'  # ViT
# pre_load_ckpt = None

# backbone_name = 'RN50'  # ViT-B/16
backbone_name = 'ViT-B/16'  # ViT-B/16
out_channels = {'RN50': 1024, 'ViT-B/16': 512}[backbone_name]

model = dict(
    type='OvarNetP',
    attribute_index_file=attribute_index_file,
    # test_content='box_given',
    test_content='box_free',
    loading_attr_emb_from_path={
        'att_path': '../pretrain/ViT16_att_emb/att_emb.pth',
        'cate_path': '../pretrain/ViT16_att_emb/cate_emb.pth',
    },
    # loading_attr_emb_from_path=None,
    box_reg='coco',  # vaw, coco, coco+vaw regression for training
    need_train_names=[
        # 'img_backbone',
        'img_neck',
        'rpn_head',
        'att_head',
        # 'prompt_category_learner',
        # 'prompt_att_learner',
        'logit_scale', 'head',
        'kd_img_align', 'kd_logit_scale',
    ],
    noneed_train_names=[],
    # img_backbone=dict(
    #     type='ResNet',
    #     depth=50,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     # frozen_stages=1,
    #     norm_cfg=dict(type='BN', requires_grad=True),
    #     norm_eval=True,
    #     style='pytorch',
    #     # load_ckpt_from='results/EXP20220809_4/epoch_50.pth'
    #     # init_cfg=dict(type='Pretrained', prefix='backbone.', map_location='cpu',
    #     #               checkpoint='results/EXP20220809_4/epoch_50.pth')
    #     # init_cfg=dict(type='Pretrained', prefix='backbone.',
    #     #               checkpoint='../pretrain/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth')
    #     init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    # ),
    img_backbone=dict(
        type='CLIPModel',
        backbone_name='RN50',
        with_attn=False,
        out_indices=[1, 2, 3, 4],
        # backbone_name='ViT-B/16',
        load_ckpt_from=pre_load_ckpt if backbone_name == 'ViT16' else None,
        precision='fp32',
    ),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        # load_ckpt_from='results/EXP20220809_4/epoch_50.pth'
        # init_cfg=dict(type='Pretrained', prefix='neck.', map_location='cpu',
        #               checkpoint='results/EXP20220809_4/epoch_50.pth')
        # init_cfg=dict(type='Pretrained', prefix='neck.', map_location='cpu',
        #               checkpoint='../pretrain/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth')
    ),
    rpn_head=dict(
        type='RPNAttrHead',
        num_convs=2,
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=2.0),
        loss_bbox=dict(type='SmoothL1Loss', loss_weight=2.0),
        # loss_bbox=dict(type='CIoULoss', loss_weight=5.0)),  # 可以修改
        # load_ckpt_from='results/EXP20220809_4/epoch_50.pth'
        # init_cfg=dict(type='Pretrained', prefix='rpn_head.', map_location='cpu',
        #               checkpoint='results/EXP20220809_4/epoch_50.pth')
    ),
    att_head=dict(
        type='ProposalEncoder',
        # out_channels=512,  # VIT/B16
        # out_channels=1024,  # R50
        out_channels=out_channels,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32, 64],
            finest_scale=28
        ),
        attribute_head=dict(
            type='TransformerAttrHead',
            in_channel=256,
            embed_dim=512,
            num_patches=14*14,
            use_abs_pos_embed=True,
            drop_rate=0.2,
            class_token=True,
            reg_token=True,
            num_encoder_layers=3,
            global_pool=False,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0),
        )
    ),
    shared_prompt_vectors=True,
    prompt_att_learner=dict(
        type='PromptAttributes',
        load_ckpt_from=pre_load_ckpt,
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
        load_ckpt_from=pre_load_ckpt,
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
    text_encoder=dict(
        type='CLIPModel',
        # backbone_name='RN50',
        # backbone_name='ViT-B/16',
        backbone_name=backbone_name,
        with_attn=False,
        out_indices=[1, 2, 3, 4],
        load_ckpt_from=pre_load_ckpt,
        precision='fp32',
    ),
    kd_model=dict(
        type='CLIPModel',
        # backbone_name='RN50',
        # backbone_name='ViT-B/16',
        backbone_name=backbone_name,
        with_attn=True,
        out_indices=[],
        load_ckpt_from=pre_load_ckpt,
        precision='fp32',
    ),

    # text_header=dict(
    #     type='TransformerEncoderHead',
    #     in_dim=1024,
    #     embed_dim=256,
    #     use_abs_pos_embed=False,
    #     drop_rate=0.05,
    #     class_token=False,
    #     num_encoder_layers=1,
    #     global_pool=False,
    # ),
    head=dict(
        type='PromptHead',
        attr_freq_file='../attributes/VAW/attr_freq_wo_sort.json',
        category_freq_file='../attributes/COCO/category_freq_wo_sort.json',
        re_weight_different_att=0.25,
        re_weight_category=1,  # 2太大了，出现cate增，att下降
        re_weight_gamma=2,
        re_weight_beta=0.995,
        # balance_unk=0.2,  # finetune
        balance_unk=0.15,
        # balance_unk=1  # gen
        kd_model_loss='t_ce+ts_ce',
        # kd_model_loss='t_ce+ts_l1',
        # kd_model_loss='t_ce+ts_l2',
        balance_kd=0.5,
        balance_teacher_loss=0.5,
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.6,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=1000,
            max_per_img=500,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=4),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.6,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=4),
        # implement in evaluation
        rcnn=dict(),
        # rcnn=dict(
        #     score_thr=0.05,
        #     nms=dict(type='nms', iou_threshold=0.5),
        #     max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    )
)

# MASK-RCNN
# rpn=dict(
#     nms_pre=1000,
#     max_per_img=1000,
#     nms=dict(type='nms', iou_threshold=0.7),
#     min_bbox_size=0),
# rcnn=dict(
#     score_thr=0.05,
#     nms=dict(type='nms', iou_threshold=0.5),
#     max_per_img=100,
#     mask_thr_binary=0.5)

# dataset settings
dataset_type = 'OvarNetAttributeDataset'
img_norm_cfg_kd = dict(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711],
    to_rgb=False
)
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53],
#     std=[58.395, 57.12, 57.375],
#     to_rgb=False
# )
img_norm_cfg = dict(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711],
    to_rgb=False
)

# Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
# img_size = (512, 512)
# img_size = (896, 896)
# img_size = (1024, 1024)
img_size = (1024, 800)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, rearrange=True, channel_order='rgb'),
    dict(
        type='Resize',
        img_scale=[(1024, 640), (1024, 800)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    # dict(type='Pad', size=img_size, center_pad=True),
    # dict(type='RandomExpandAndCropBox', expand_range=(0.95, 1.2), crop_range=(0.85, 1), prob=0.6),
    dict(type='ImageToTensor', keys=['img']),
    # dict(type='DefaultFormatBundle'),
    dict(type='ToTensor', keys=['gt_bboxes', 'gt_labels', 'instance_sources']),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'instance_sources'])
]

kd_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, rearrange=True, channel_order='rgb'),
    dict(type='ScaleCrop', scale_range=[0.0, 0.1]),
    dict(type='RandomCrop', crop_size=[0.9, 0.9], crop_type='relative_range'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=(224, 224), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg_kd),
    dict(type='Pad', size=(224, 224), center_pad=True),
    dict(type='ImageToTensor', keys=['img']),
]

test_box_given_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True,  rearrange=True, channel_order='rgb'),
    dict(type='MultiScaleFlipAug',
         img_scale=img_size,
         flip=False,
         transforms=[
            dict(type='Resize', img_scale=img_size, keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            # dict(type='Pad', size=img_size,  center_pad=True),
            # dict(type='RandomExpandAndCropBox', expand_range=(0.95, 1.2), crop_range=(0.9, 1)),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'gt_bboxes'])
        ]
    )
]

test_box_free_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True,  rearrange=True, channel_order='rgb'),
    dict(type='MultiScaleFlipAug',
         img_scale=img_size,
         flip=False,
         transforms=[
            dict(type='Resize', img_scale=img_size, keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            # dict(type='Pad', size=img_size,  center_pad=True),
            # dict(type='RandomExpandAndCropBox', expand_range=(0.95, 1.2), crop_range=(0.9, 1)),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]
    )
]

# find_unused_parameters = True
samples_per_gpu = 16
data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=4,
    # samples_per_gpu=4,
    # workers_per_gpu=0,
    persistent_workers=True,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        dataset_split='train',
        attribute_index_file=attribute_index_file,
        dataset_names=['coco', 'vaw'],
        test_mode=False,
        pipeline=train_pipeline,
        kd_pipeline=kd_pipeline,
        # kd_pipeline=None,
        dataset_balance=True
    ),
    val=dict(
        samples_per_gpu=samples_per_gpu,
        type=dataset_type,
        data_root=data_root,
        dataset_split='test',
        attribute_index_file=attribute_index_file,
        dataset_names=['coco', 'vaw'],
        test_mode=True,
        mult_proposal_score=False,
        test_content='box_given',
        pipeline=test_box_given_pipeline,
    ),
    test=dict(
        samples_per_gpu=20,
        type=dataset_type,
        data_root=data_root,
        dataset_split='test',
        attribute_index_file=dict(
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
        ),
        dataset_names=['coco', 'vaw'],
        # dataset_names=['ovadcate', 'ovadattr'],
        test_mode=True,
        mult_proposal_score=False,
        # test_content='box_given',
        # pipeline=test_box_given_pipeline,
        test_content='box_free',
        pipeline=test_box_free_pipeline,
    )
)
# #
# optimizer
# optimizer = dict(
#     constructor='SubModelConstructor',
#     sub_model={
#         'prompt_learner': {},
#         # 'text_encoder': {'lr_mult': 0.01},
#         # 'image_encoder': {'lr_mult': 0.01},
#         'neck': {}, 'roi_head': {},
#         'kd_logit_scale': {}, 'kd_img_align': {},
#         'bbox_head': {}, 'logit_scale': {},
#         # 'text_encoder': {'lr_mult': 0.01},
#         # 'kd_model': {'lr_mult': 0.1},
#         # 'kd_logit_scale': {}, 'kd_img_align': {},
#         # 'prompt_learner': {},
#     },
#     type='SGD',
#     lr=0.01,
#     momentum=0.9,
#     weight_decay=0.0005
# )

# # optimizer
optimizer = dict(
    constructor='SubModelConstructor',
    sub_model={
        # 'img_backbone': {'lr_mult': 0.1},
        'img_neck': {},
        'rpn_head': {},
        'att_head': {},
        # 'prompt_att_learner': {'lr_mult': 0.1},
        # 'prompt_category_learner': {},
        'logit_scale': {}, 'head': {},
        'kd_img_align': {}, 'kd_logit_scale': {}
        },
    type='AdamW',
    lr=1e-4,
    weight_decay=1e-3
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# # learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.1,
    step=[30, 40])

# lr_config = dict(
#     policy='CosineAnnealing',
#     by_epoch=False,
#     min_lr_ratio=1e-2,
#     warmup='linear',
#     warmup_ratio=1e-3,
#     warmup_iters=1,
#     warmup_by_epoch=True)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=50)
evaluation = dict(
    interval=5, metric='mAP',
    nms_cfg=dict(
        type='nms',
        class_agnostic=False,
        iou_threshold=0.55,
        proposal_score_thr=0.2,
        score_thr=0.4,
        max_num=100))
# rcnn=dict(
#     score_thr=0.05,
#     nms=dict(type='nms', iou_threshold=0.5),
#     max_per_img=100,
#     mask_thr_binary=0.5)
load_from = None
# resume_from = 'results/EXP20220905_0/latest.pth'
resume_from = None