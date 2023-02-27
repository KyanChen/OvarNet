import torch
from mmcv.runner import BaseModule, get_dist_info

from ..builder import BACKBONES

from .clip import _MODELS, _download, build_model

from collections import OrderedDict


@BACKBONES.register_module()
class CLIPModel(BaseModule):
    def __init__(self,
                 backbone_name='RN50',
                 with_attn=True,
                 out_indices=(1, 2, 3, 4),
                 load_ckpt_from=None,
                 precision='fp16',
                 pretrained=None,
                 init_cfg=None):
        super(CLIPModel, self).__init__(init_cfg)
        if backbone_name not in _MODELS.keys():
            raise KeyError(f'invalid backbone_name {backbone_name} for CLIPModel')

        assert precision in ["fp16", "fp32", "amp"]
        # import pdb
        # pdb.set_trace()
        rank, world_size = get_dist_info()
        url = _MODELS[backbone_name]
        if load_ckpt_from is None:
            load_ckpt_from = _download(url)
        try:
            if rank == 0:
                print(f'load clip ckpt from {load_ckpt_from}')
            # loading JIT archive
            model = torch.jit.load(load_ckpt_from, map_location="cpu").eval()
            new_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(load_ckpt_from, map_location="cpu")['state_dict']
            new_dict = OrderedDict()
            for k, v in state_dict.items():
                k = k.replace('image_encoder', 'visual')
                k = k.replace('text_encoder.', '')
                new_dict[k] = v

            load_ckpt_from = _download(url)
            model = torch.jit.load(load_ckpt_from, map_location="cpu").eval()
            dict_tmp = model.state_dict()
            for k, v in dict_tmp.items():
                if k in new_dict.keys():
                    continue
                else:
                    if rank == 0:
                        print(f'load state dict: {k} from origin dict')
                    new_dict[k] = v
        # import pdb
        # pdb.set_trace()
        self.model = build_model(new_dict, with_attn, out_indices=out_indices)

        if precision == "fp32" or precision == "amp":
            # CLIP's default precision is fp16
            self.model = self.model.float()


@BACKBONES.register_module()
class TextEncoder(BaseModule):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding  # 77 512
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    # def forward(self, prompts, tokenized_prompts):
    #     x = prompts + self.positional_embedding.type(self.dtype)
    #     x = x.permute(1, 0, 2)  # NLD -> LND
    #     x = self.transformer(x)
    #     x = x.permute(1, 0, 2)  # LND -> NLD
    #
    #     x = self.ln_final(x).type(self.dtype)  # 620 77 512
    #
    #     # x.shape = [batch_size, n_ctx, transformer.width]
    #     # take features from the eot embedding (eot_token is the highest number in each sequence)
    #     x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
    #     # x: 620x1024
    #     return x

    def forward(self, prompts, eot_index):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_final(x).type(self.dtype)  # 620 77 512

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), eot_index.argmax(dim=-1)] @ self.text_projection
        x = x[torch.arange(x.shape[0]), eot_index] @ self.text_projection
        # x: 620x1024
        return x

