from functools import partial
import torch
import torch.nn as nn
import timm.models.vision_transformer

from ..builder import BACKBONES
from einops import rearrange

@BACKBONES.register_module()
class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    archs = {
        'vit_base_patch16': dict(
            patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        ),
        'vit_large_patch16': dict(
            patch_size=16, embed_dim=1024, depth=24, num_heads=16,
            mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        ),
        'vit_huge_patch14': dict(
            patch_size=14, embed_dim=1280, depth=32, num_heads=16,
            mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
    }

    def __init__(self, arch, load_pretrain=None, global_pool=False, **kwargs):
        arch = self.archs[arch]
        arch.update(kwargs)
        arch['weight_init'] = 'skip'
        super(VisionTransformer, self).__init__(**arch)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        self.load_pretrain = load_pretrain

    def load_pretrain_model(self):
        if self.load_pretrain is not None:
            # import pdb
            # pdb.set_trace()
            checkpoint = torch.load(self.load_pretrain, map_location='cpu')
            print("Load pre-trained vit checkpoint from: %s" % self.load_pretrain)
            if 'model' in checkpoint.keys():
                checkpoint_model = checkpoint['model']
            else:
                checkpoint_model = checkpoint

            state_dict = self.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            # interpolate position embedding
            interpolate_pos_embed(self, checkpoint_model)

            # load pre-trained model
            msg = self.load_state_dict(checkpoint_model, strict=False)
            if self.global_pool:
                assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
            else:
                assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

    def init_weights(self, mode=''):
        self.load_pretrain_model()

    def forward_features(self, x):
        B = x.shape[0]
        H = x.shape[-2]
        x = self.patch_embed(x)  # 128x196x768

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        # import pdb
        # pdb.set_trace()
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
            last_f_map = x[:, 1:]
            last_f_map = rearrange(last_f_map, 'B (H W) C -> B C H W', H=H//16)

        return outcome, last_f_map, None

    def forward(self, x):
        return self.forward_features(x)


def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

