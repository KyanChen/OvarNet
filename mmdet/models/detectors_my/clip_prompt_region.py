import gc
import json
from collections import OrderedDict

import torch
from torch import nn
from einops import rearrange

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from ..detectors.base import BaseDetector
import warnings
from mmcv.runner import BaseModule, get_dist_info
import torch.distributed as dist


@DETECTORS.register_module()
class CLIP_Prompter_Region(BaseModule):
    def __init__(self,
                 classname_path,
                 need_train_names,
                 noneed_train_names,
                 img_backbone,
                 img_neck,
                 img_head,
                 prompt_learner,
                 text_encoder,
                 head,
                 kd_model=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(CLIP_Prompter_Region, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')

        classname_maps = json.load(open(classname_path))
        classnames = list(classname_maps.keys())

        if kd_model:
            model_tmp = build_backbone(kd_model).model
            self.kd_model = model_tmp.visual.eval()
            self.kd_logit_scale = nn.Parameter(model_tmp.logit_scale.data)
            self.kd_img_align = nn.Linear(1024, 1024)

        # if text_header:
        #     self.text_header = build_head(text_header)
        if 'CLIPModel' in [img_backbone['type'], text_encoder['type']]:
            if img_backbone['type'] == 'CLIPModel':
                clip_config = img_backbone
            else:
                clip_config = text_encoder
            clip_model = build_backbone(clip_config).model

        self.with_clip_img_backbone = False
        if img_backbone['type'] == 'CLIPModel':
            self.img_backbone = clip_model.visual.eval()
            self.with_clip_img_backbone = True
        else:
            load_ckpt_from = img_backbone.pop('load_ckpt_from', None)
            self.img_backbone = build_backbone(img_backbone)
            if load_ckpt_from is not None:
                state_dict = torch.load(load_ckpt_from, map_location="cpu")['state_dict']
                new_dict = OrderedDict()
                for k, v in state_dict.items():
                    k = k.replace('backbone.', '')
                    new_dict[k] = v

                missing_keys, unexpected_keys = self.img_backbone.load_state_dict(new_dict, strict=False)
                print('load img_backbone: ')
                print('missing_keys: ', missing_keys)
                print('unexpected_keys: ', unexpected_keys)
                print()

        if text_encoder['type'] == 'CLIPModel':
            self.text_encoder = build_backbone(
                dict(type='TextEncoder', clip_model=clip_model)
            )
            self.logit_scale = nn.Parameter(clip_model.logit_scale.data)

        prompt_learner.update(dict(classnames=classnames, clip_model=clip_model))
        self.prompt_learner = build_backbone(prompt_learner)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        if img_neck is not None:
            load_ckpt_from = img_neck.pop('load_ckpt_from', None)
            self.img_neck = build_neck(img_neck)
            if load_ckpt_from is not None:
                state_dict = torch.load(load_ckpt_from, map_location="cpu")['state_dict']
                new_dict = OrderedDict()
                for k, v in state_dict.items():
                    k = k.replace('neck.', '')
                    new_dict[k] = v

                missing_keys, unexpected_keys = self.img_neck.load_state_dict(new_dict, strict=False)
                print('load img_neck: ')
                print('missing_keys: ', missing_keys)
                print('unexpected_keys: ', unexpected_keys)
                print()

        if img_head is not None:
            self.img_head = build_head(img_head)

        self.head = build_head(head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.need_train_names = need_train_names
        self.noneed_train_names = noneed_train_names
        self._set_grad(need_train_names, noneed_train_names)

    def _set_grad(self, need_train_names: list, noneed_train_names: list):
        for name, param in self.named_parameters():
            flag = False
            for need_train_name in need_train_names:
                if need_train_name in name:
                    flag = True
            for noneed_train_name in noneed_train_names:
                if noneed_train_name in name:
                    flag = False
            param.requires_grad_(flag)

        not_specific_names = []
        for name, param in self.named_parameters():
            flag_find = False
            for specific_name in need_train_names + noneed_train_names:
                if specific_name in name:
                    flag_find = True
            if not flag_find:
                not_specific_names.append(name)

        _rank, _word_size = get_dist_info()
        if _rank == 0:
            print(f"Turning off gradients for names: {noneed_train_names}")
            print(f"Turning on gradients for names: {need_train_names}")
            print(f"Turning off gradients for not specific names: {not_specific_names}")

    def train(self, mode=True):
        self.training = mode
        for name, module in self.named_children():
            flag = False
            for need_train_name in self.need_train_names:
                if need_train_name in name:
                    flag = True
            if flag:
                module.train(mode)
            else:
                module.eval()
        return self

    def forward(self, img, img_metas, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def train_step(self, data, optimizer):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def _parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        # If the loss_vars has different length, GPUs will wait infinitely
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' +
                       f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()))
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def val_step(self, data, optimizer=None):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    @property
    def with_neck(self):
        return hasattr(self, 'img_neck') and self.img_neck is not None

    @property
    def with_kd_model(self):
        return hasattr(self, 'kd_model') and self.kd_model is not None

    def forward_train(self,
                      img,
                      img_metas,
                      proposals,
                      gt_labels,
                      **kwargs
                      ):
        if self.with_clip_img_backbone:
            image_features, final_map, img_f_maps = self.img_backbone(img)  # 2x1024
        else:
            img_f_maps = self.img_backbone(img)
        if self.with_neck:
            img_f_maps = self.img_neck(img_f_maps)
        proposal_features, bbox_feats = self.img_head(img_f_maps, proposals)

        prompts = self.prompt_learner()  # 620x77x512
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        proposal_features = proposal_features / proposal_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        extra_info = {'proposal_features': proposal_features}
        if "img_crops" in kwargs and self.with_kd_model:
            img_crops = kwargs.get('img_crops', None)
            img_crops = torch.cat(img_crops, dim=0)
            with torch.no_grad():
                img_crop_features, _, _ = self.kd_model(img_crops)
            img_crop_features = self.kd_img_align(img_crop_features)
            img_crop_features = img_crop_features / img_crop_features.norm(dim=-1, keepdim=True)
            extra_info['img_crop_features'] = img_crop_features
            kd_logit_scale = self.kd_logit_scale.exp()
            kd_logits = kd_logit_scale * img_crop_features @ text_features.t()  # 2x620
            extra_info['kd_logits'] = kd_logits

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * proposal_features @ text_features.t()  # 2x620

        gt_labels = torch.cat(gt_labels, dim=0)
        losses = self.head.forward_train(logits, img_metas, gt_labels, **extra_info)

        return losses

    def forward_test(self, imgs, img_metas, **kwargs):
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)

    def simple_test(self,
                    img, img_metas, proposals,
                    rescale=False, **kwargs):

        if self.with_clip_img_backbone:
            image_features, final_map, img_f_maps = self.img_backbone(img)  # 2x1024
        else:
            img_f_maps = self.img_backbone(img)
        per_img_proposals = [len(x) for x in proposals]
        if self.with_neck:
            img_f_maps = self.img_neck(img_f_maps)
        proposal_features, bbox_feats = self.img_head(img_f_maps, proposals)

        prompts = self.prompt_learner()  # 620x77x512
        tokenized_prompts = self.tokenized_prompts

        text_features = self.text_encoder(prompts, tokenized_prompts)  # torch.Size([620, 1024])
        proposal_features = proposal_features / proposal_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # extra_info = {'proposal_features': proposal_features}
        # if "img_crops" in kwargs and hasattr(self, 'kd_model'):
        #     img_crops = kwargs.get('img_crops', None)
        #     img_crops = torch.cat(img_crops, dim=0)
        #     # with torch.no_grad():
        #     img_crop_features, _, _ = self.kd_model(img_crops)
        #     img_crop_features = self.kd_img_align(img_crop_features)
        #     img_crop_features = img_crop_features / img_crop_features.norm(dim=-1, keepdim=True)
        #     extra_info['img_crop_features'] = img_crop_features
        #     kd_logit_scale = self.kd_logit_scale.exp()
        #     kd_logits = kd_logit_scale * img_crop_features @ text_features.t()  # 2x620

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * proposal_features @ text_features.t()  # 2x620
        logits = torch.split(logits, per_img_proposals, dim=0)

        pred = [x.detach().cpu().numpy() for x in logits]
        return pred
