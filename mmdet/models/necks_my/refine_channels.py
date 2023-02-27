# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import NECKS


@NECKS.register_module()
class RefineChannel(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(RefineChannel, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.ops = nn.ModuleList()
        for i in in_channels:
            self.ops.append(
                nn.Conv2d(i, out_channels, kernel_size=1)
            )
        if num_outs > len(self.in_channels):
            self.ops.append(
                nn.Conv2d(in_channels[-1], out_channels, kernel_size=3, stride=2, padding=1)
            )

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        outs = []
        for i, layer in enumerate(self.ops[:len(self.in_channels)]):
            outs.append(layer(inputs[i]))
        for i, layer in enumerate(self.ops[len(self.in_channels):]):
            outs.append(layer(inputs[-1]))

        return tuple(outs)
