from mmcv.runner import BaseModule
from ..builder import BACKBONES
from mmcv import ConfigDict
import torch
from typing import Optional

from .ofa import (ofa_tiny_architecture,
                  ofa_medium_architecture,
                  ofa_huge_architecture,
                  ofa_base_architecture,
                  ofa_large_architecture, OFAModel)
from fairseq.data import Dictionary
from fairseq.data.encoders import build_bpe
import json
from utils_my import Trie


@BACKBONES.register_module()
class OFA(BaseModule):
    __arch_func = {
        'ofa_tiny': ofa_tiny_architecture,
        'ofa_medium': ofa_medium_architecture,
        'ofa_huge': ofa_huge_architecture,
        'ofa_base': ofa_base_architecture,
        'ofa_large': ofa_large_architecture
    }

    def __init__(self,
                 ofa_name,
                 task,
                 model_config=dict(),
                 model_cfg=dict()
                 ):
        super().__init__()
        self.cfg = self.get_default_args()
        self.cfg.update(model_cfg)
        self.cfg.update(model_config)
        self.__arch_func[ofa_name](self.cfg)

        self.model = OFAModel.build_model(self.cfg, task)


    def get_default_args(self):
        args = ConfigDict()
        args.bpe_dir = None
        args.max_source_positions = 1024
        args.max_target_positions = 1024
        args.max_src_length = 128
        args.max_tgt_length = 30
        args.code_dict_size = 8192
        args.patch_image_size = 480
        args.num_bins = 1000
        args.constraint_range = None
        return args
