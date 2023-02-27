checkpoint_config = dict(interval=30)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]
# custom_hooks = None

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
    type='DETR',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    bbox_head=dict(
        type='DETRHead',
        # num_classes=80,
        num_classes=10,
        in_channels=2048,
        transformer=dict(
            type='Transformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1)
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DetrTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1),
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True),
        loss_cls=dict(
            type='CrossEntropyLoss',
            bg_cls_weight=0.1,
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='ClassificationCost', weight=1.),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
    test_cfg=dict(max_per_img=100)
)
# dataset settings
dataset_type = 'NWPUDataset'
# data_root = 'D:/Dataset/NWPU VHR-10 dataset'
data_root = '/data/kyanchen/prompt1/data'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
    to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[
            [
                dict(
                    type='Resize',
                    img_scale=[
                        (480, 900), (512, 900), (544, 900), (576, 900),
                        (608, 900), (640, 900), (672, 900), (704, 900),
                        (736, 900), (768, 900), (800, 900)
                    ],
                    multiscale_mode='value',
                    keep_ratio=True)
            ],
            [
                dict(
                    type='Resize',
                    img_scale=[(400, 900), (500, 900), (600, 900)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 500),
                    allow_negative_crop=True),
                dict(
                    type='Resize',
                    img_scale=[
                        (480, 900), (512, 900), (544, 900),
                        (576, 900), (608, 900), (640, 900),
                        (672, 900), (704, 900), (736, 900),
                        (768, 900), (800, 900)
                    ],
                    multiscale_mode='value',
                    override=True,
                    keep_ratio=True
                )
            ]
        ]
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(600, 900),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=24,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + '/NWPU_train.json',
        img_prefix=data_root + '/PositiveTrain',
        pipeline=train_pipeline),
    val=dict(
        samples_per_gpu=4,
        type=dataset_type,
        ann_file=data_root + '/NWPU_val.json',
        img_prefix=data_root + '/PositiveEvaluation',
        pipeline=test_pipeline),
    test=dict(
        samples_per_gpu=4,
        type=dataset_type,
        ann_file=data_root + '/NWPU_val.json',
        img_prefix=data_root + '/PositiveEvaluation',
        pipeline=test_pipeline)
)
evaluation = dict(interval=10, metric='bbox')
test = dict(interval=10, metric=['bbox'], mode='test', areaRng=[0, 32, 96])

# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(grad_clip=dict(max_norm=100, norm_type=2))

# learning policy
# lr_config = dict(policy='step', step=[100])
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=1e-2,
    warmup='linear',
    warmup_ratio=1e-3,
    warmup_iters=5,
    warmup_by_epoch=True)

runner = dict(type='EpochBasedRunner', max_epochs=500)

load_from = '../pretrain/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth'
resume_from = None