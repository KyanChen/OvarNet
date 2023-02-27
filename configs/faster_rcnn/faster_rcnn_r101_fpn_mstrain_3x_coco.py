_base_ = 'Op1_2_class_agnostic_rpn_mstrain.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
