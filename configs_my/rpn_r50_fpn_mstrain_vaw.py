checkpoint_config = dict(interval=2)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = None

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

model = dict(
    type='FasterRCNNRPN',
    need_train_names=[
        'backbone', 'neck', 'rpn_head'
    ],
    noneed_train_names=[
        'backbone.layer1', 'backbone.conv1', 'backbone.bn1'
    ],
    # backbone=dict(
    #     type='ResNet',
    #     depth=50,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     frozen_stages=1,
    #     norm_cfg=dict(type='BN', requires_grad=True),
    #     norm_eval=True,
    #     style='pytorch',
    #     init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
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
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        num_convs=3,
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
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
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
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        # rcnn=dict(
        #     assigner=dict(
        #         type='MaxIoUAssigner',
        #         pos_iou_thr=0.5,
        #         neg_iou_thr=0.5,
        #         min_pos_iou=0.5,
        #         match_low_quality=False,
        #         ignore_iof_thr=-1),
        #     sampler=dict(
        #         type='RandomSampler',
        #         num=512,
        #         pos_fraction=0.25,
        #         neg_pos_ub=-1,
        #         add_gt_as_proposals=True),
        #     pos_weight=-1,
        #     debug=False)
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        # rcnn=dict(
        #     score_thr=0.05,
        #     nms=dict(type='nms', iou_threshold=0.5),
        #     max_per_img=100)

        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=False)

img_norm_cfg = dict(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711],
    to_rgb=False
)

# In mstrain 3x config, img_scale=[(1333, 640), (1333, 800)],
# multiscale_mode='range'
train_pipeline = [
    dict(type='LoadImageFromFile', rearrange=True, channel_order='rgb'),
    # dict(type='LoadImageFromFile', channel_order='rgb'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1024, 640), (1024, 800)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    # dict(type='LoadImageFromFile', channel_order='rgb'),
    dict(type='LoadImageFromFile', rearrange=True, channel_order='rgb'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# dataset_type = 'CocoRPNDataset'
dataset_type = 'VAWRPNDataset'

# data_root = '/data1/kyanchen/DetFramework/data/COCO/'
# data_root = '/data1/kyanchen/prompt/data'
data_root = '/data/kyanchen/Data'
# data_root = '/data/kyanchen/prompt/data'

samples_per_gpu = 20
data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root+'/train2017',
        ann_file=data_root+'/annotations/instances_train2017.json',
        pipeline=train_pipeline,
        test_mode=False
    ),
    val=dict(
        samples_per_gpu=samples_per_gpu,
        type=dataset_type,
        data_root=data_root+'/val2017',
        ann_file=data_root + '/annotations/instances_val2017.json',
        pipeline=test_pipeline,
        test_mode=True
    ),
    test=dict(
        samples_per_gpu=samples_per_gpu,
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        pattern='test',
        test_mode=True
    )
)
evaluation = dict(interval=5, metric='proposal_fast')

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy
# Experiments show that using step=[9, 11] has higher performance
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[40, 50])
runner = dict(type='EpochBasedRunner', max_epochs=60)
load_from = None
resume_from = None