checkpoint_config = dict(interval=20)
# yapf:disable
log_config = dict(
    interval=1,
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
find_unused_parameters = True
# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)

# attribute_id_map = '../attributes/COCO/attribute_id_map.json'
# attribute_id_map = '/Users/kyanchen/Code/CLIP_Prompt/attributes/COCO/attribute_id_map.json'
# attribute_id_map = 'I:/CodeRep/CLIP_Prompt/attributes/COCO/attribute_id_map.json'
attribute_id_map = '/data/kyanchen/prompt/attributes/COCO/attribute_id_map.json'
# model settings
model = dict(
    type='MaskRCNNCLIP',
    with_proposal_ann=True,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
        ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
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
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='RoIHeadWoMask',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=None,
        mask_head=None
    ),
    proposal_encoder=dict(
        type='ProposalEncoder',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32, 64]
        ),
        shared_head=dict(
            type='ResLayer',
            depth=50,
            stage=3,
            stride=1,
            norm_eval=False,
            inplanes=256,
            planes=128,
        ),
        in_channels=512,
        out_channels=1024,
    ),
    attribute_encoder=dict(
        type='AttributeEncoder',
        attribute_id_map=attribute_id_map,
        n_ctx=16,
        prompt_num=8,
        class_token_position='mid',
        context_length=32,
        model_dim=512,
        out_channels=1024,
    ),
    attribute_pred_head=dict(
        type='AttributePredHead',
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0
        )
    ),
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
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            # mask_thr_binary=0.5
        )
    )
)


# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# In mstrain 3x config, img_scale=[(1333, 640), (1333, 800)],
# multiscale_mode='range'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_label=False),
    dict(
        type='Resize',
        # img_scale=[(1333, 640), (1333, 800)],
        # multiscale_mode='range',
        img_scale=(1333, 800),
        keep_ratio=True),
    # dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
    dict(type='ToTensor', keys=['attrs']),
    dict(type='Collect', keys=['img', 'attrs', 'gt_bboxes']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_label=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            # dict(type='ToTensor', keys=['attrs']),
            dict(type='Collect', keys=['img', 'gt_bboxes']),
        ])
]

# Use RepeatDataset to speed up training
caption_root = '/data/kyanchen/prompt/data/COCO'
# caption_root = '/Users/kyanchen/Code/CLIP_Prompt/captions/COCO'
# category_id_map = '../objects/MSCOCO/category_id_map.json'
# category_id_map = '/Users/kyanchen/Code/CLIP_Prompt/objects/MSCOCO/category_id_map.json'
category_id_map = '/data/kyanchen/prompt/objects/MSCOCO/category_id_map.json'
attributes_file = '/data/kyanchen/prompt/data/COCO/attributes_2014.pkl'

dataset_type = 'CocoCLIPAnnDataset'
img_root = '/data/kyanchen/prompt/data/COCO'
data = dict(
    train_dataloader=dict(shuffle=False),
    samples_per_gpu=10,
    workers_per_gpu=4,
    persistent_workers=True,
    train=dict(
        type=dataset_type,
        attributes_file=attributes_file,
        annotations_file=caption_root+'/annotations/instances_train2014.json',
        pipeline=train_pipeline,
        attribute_id_map=attribute_id_map,
        img_prefix=caption_root+'/train2014',
        att_split='train2014',
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        attributes_file='D:/Dataset/COCO/attributes_2014.pkl',
        annotations_file="D:/Dataset/COCO/instances_val2014.json",
        pipeline=test_pipeline,
        attribute_id_map='I:/CodeRep/CLIP_Prompt/attributes/COCO/attribute_id_map.json',
        img_prefix="D:/Dataset/COCO/val2014",
        test_mode=True,
    ),
    test=dict(
        samples_per_gpu=16,
        is_replace_ImageToTensor=True,
        type=dataset_type,
        attributes_file=attributes_file,
        annotations_file=caption_root + '/annotations/instances_val2014.json',
        pipeline=test_pipeline,
        attribute_id_map=attribute_id_map,
        img_prefix=caption_root + '/val2014',
        att_split='val2014',
        test_mode=True,
    )
)
evaluation = dict(interval=5, metric=['bbox'])

# optimizer
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)

optimizer = dict(
    constructor='SubModelConstructor',
    sub_model=['proposal_encoder', 'attribute_encoder', 'attribute_pred_head'],
    type='AdamW',
    lr=1e-4
)
optimizer_config = dict(grad_clip=None)

# learning policy
# Experiments show that using step=[9, 11] has higher performance
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    # step=[50, 80]
    step=[300]
)
runner = dict(type='EpochBasedRunner', max_epochs=500)
# load_from = '/data/kyanchen/prompt/pretrain/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_20210524_201154-21b550bb.pth'
# load_from = 'D:/Dataset/COCO/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_20210524_201154-21b550bb.pth'
load_from = None
resume_from = None