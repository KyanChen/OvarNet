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
class LSAOvarNet(BaseModule):
    def __init__(self,
                 attribute_index_file,
                 need_train_names,
                 noneed_train_names,
                 img_backbone,
                 img_neck,
                 prompt_catephrase_learner,
                 att_head,
                 shared_prompt_vectors,
                 head,
                 test_content,
                 text_emb_from,
                 prompt_att_learner=None,
                 box_reg=None,  # vaw, coco, vaw+coco RPN是否包含属性预测的内容
                 prompt_category_learner=None,
                 rpn_head=None,
                 text_encoder=None,
                 kd_model=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(LSAOvarNet, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
        self.box_reg = box_reg

        self.att2id = {}
        if 'att_file' in attribute_index_file.keys():
            file = attribute_index_file['att_file']
            att2id = json.load(open(file, 'r'))
            att_group = attribute_index_file['att_group']
            if att_group in ['common1', 'common2', 'common', 'rare']:
                self.att2id = att2id[att_group]
            elif att_group == 'common1+common2':
                self.att2id.update(att2id['common1'])
                self.att2id.update(att2id['common2'])
            elif att_group == 'common+rare':
                self.att2id.update(att2id['common'])
                self.att2id.update(att2id['rare'])
            elif att_group == 'base+novel':
                self.att2id.update(att2id['base'])
                self.att2id.update(att2id['novel'])
                # self.att_seen_unseen['seen'] = list(att2id['base'].keys())
                # self.att_seen_unseen['unseen'] = list(att2id['novel'].keys())
            else:
                raise NameError
        self.category2id = {}
        if 'category_file' in attribute_index_file.keys():
            file = attribute_index_file['category_file']
            category2id = json.load(open(file, 'r'))
            att_group = attribute_index_file['category_group']
            if att_group in ['common1', 'common2', 'common', 'rare']:
                self.category2id = category2id[att_group]
            elif att_group == 'common1+common2':
                self.category2id.update(category2id['common1'])
                self.category2id.update(category2id['common2'])
            elif att_group == 'common+rare':
                self.category2id.update(category2id['common'])
                self.category2id.update(category2id['rare'])
            else:
                raise NameError
        self.att2id = {k: v - min(self.att2id.values()) for k, v in self.att2id.items()}
        self.category2id = {k: v - min(self.category2id.values()) for k, v in self.category2id.items()}

        rank, world_size = get_dist_info()

        self.with_clip_img_backbone = False
        if rank == 0:
            print('build img backbone ', img_backbone['type'])
        if img_backbone['type'] == 'CLIPModel':
            clip_model = build_backbone(img_backbone).model
            self.img_backbone = clip_model.visual.eval()
            self.with_clip_img_backbone = True
            self.logit_scale = nn.Parameter(clip_model.logit_scale.data)
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
                if rank == 0:
                    print('load img_backbone: ')
                    print('missing_keys: ', missing_keys)
                    print('unexpected_keys: ', unexpected_keys)
                    print()

        if prompt_catephrase_learner is not None:
            prompt_catephrase_learner.update(
                dict(clip_model=clip_model)
            )
            self.prompt_catephrase_learner = build_backbone(prompt_catephrase_learner)

        self.text_encoder = build_backbone(
            dict(
                type='TextEncoder',
                clip_model=clip_model
            )
        )

        self.text_emb_from = text_emb_from
        print(f'load attribute emb from {text_emb_from}')
        self.text_features = torch.load(text_emb_from).cpu()

        if text_emb_from is not None and prompt_att_learner is not None:
            if rank == 0:
                print('build prompt_att_learner')
            assert len(self.att2id)
            clip_model = build_backbone(text_encoder).model
            prompt_att_learner.update(
                dict(attribute_list=list(self.att2id.keys()),
                     clip_model=clip_model,
                     self_name='prompt_att_learner'
                     )
            )
            self.prompt_att_learner = build_backbone(prompt_att_learner)

        if prompt_category_learner is not None:
            if rank == 0:
                print('build prompt_att_learner')
            assert len(self.category2id)
            prompt_category_learner.update(
                dict(attribute_list=list(self.category2id.keys()),
                     clip_model=clip_model,
                     self_name='prompt_category_learner'
                     )
            )
            if shared_prompt_vectors:
                prompt_category_learner.update(
                    dict(shared_prompt_vectors=self.prompt_att_learner.prompt_vectors)
                )
            self.prompt_category_learner = build_backbone(prompt_category_learner)

        if img_neck is not None:
            load_ckpt_from = img_neck.pop('load_ckpt_from', None)
            if rank == 0:
                print('build img neck')
            self.img_neck = build_neck(img_neck)
            if load_ckpt_from is not None:
                state_dict = torch.load(load_ckpt_from, map_location="cpu")['state_dict']
                new_dict = OrderedDict()
                for k, v in state_dict.items():
                    k = k.replace('neck.', '')
                    new_dict[k] = v

                missing_keys, unexpected_keys = self.img_neck.load_state_dict(new_dict, strict=False)
                if rank == 0:
                    print('load img_neck: ')
                    print('missing_keys: ', missing_keys)
                    print('unexpected_keys: ', unexpected_keys)
                    print()

        if att_head is not None:
            if rank == 0:
                print('build attribute head')
            self.att_head = build_head(att_head)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            if rank == 0:
                print('build rpn head')
            load_ckpt_from = rpn_head_.pop('load_ckpt_from', None)
            self.rpn_head = build_head(rpn_head_)
            if load_ckpt_from is not None:
                state_dict = torch.load(load_ckpt_from, map_location="cpu")['state_dict']
                new_dict = OrderedDict()
                for k, v in state_dict.items():
                    k = k.replace('rpn_head.', '')
                    new_dict[k] = v

                missing_keys, unexpected_keys = self.rpn_head.load_state_dict(new_dict, strict=False)
                if rank == 0:
                    print('load rpn head: ')
                    print('missing_keys: ', missing_keys)
                    print('unexpected_keys: ', unexpected_keys)
                    print()

        head['attribute_index_file'] = attribute_index_file
        if rank == 0:
            print('build head')
        self.head = build_head(head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.test_content = test_content
        assert self.test_content in ['box_given', 'box_free']

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
            not_specific_names = [x.split('.')[0] for x in not_specific_names]
            not_specific_names = set(not_specific_names)
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
                      gt_bboxes,
                      gt_labels,
                      dataset_type,
                      phases,
                      **kwargs
                      ):

        if self.with_clip_img_backbone:
            image_features, final_map, img_f_maps = self.img_backbone(img)  # 2x1024
        else:
            img_f_maps = self.img_backbone(img)
        img_f_maps = self.img_neck(img_f_maps)
        losses = dict()

        # if self.box_reg == 'coco+vaw':
        #     seen_mask = (dataset_type == 0) | (dataset_type == 1)
        #     if torch.any(seen_mask):  # 存在coco seen 目标
        #
        #         img_rpn = [x[seen_mask, ...] for x in img_f_maps]
        #         boxes_rpn = [x for idx, x in enumerate(gt_bboxes) if seen_mask[idx]]
        #         img_metas_rpn = [x for idx, x in enumerate(img_metas) if seen_mask[idx]]
        #         rpn_losses = self.rpn_head.forward_train(img_rpn,
        #                                                  img_metas_rpn,
        #                                                  boxes_rpn,
        #                                                  gt_labels=None,
        #                                                  gt_bboxes_ignore=None,
        #                                                  proposal_cfg=None,
        #                                                  **kwargs)
        #     else:
        #         rpn_losses = dict(loss_rpn_cls=torch.tensor(0.).to(img.device),
        #                           loss_rpn_bbox=torch.tensor(0.).to(img.device))
        # elif self.box_reg == 'coco':
        #     coco_seen_mask = dataset_type == 0
        #     if torch.any(coco_seen_mask):  # 存在coco seen 目标
        #
        #         img_rpn = [x[coco_seen_mask, ...] for x in img_f_maps]
        #         boxes_rpn = [x for idx, x in enumerate(gt_bboxes) if coco_seen_mask[idx]]
        #         img_metas_rpn = [x for idx, x in enumerate(img_metas) if coco_seen_mask[idx]]
        #         rpn_losses = self.rpn_head.forward_train(img_rpn,
        #                                                  img_metas_rpn,
        #                                                  boxes_rpn,
        #                                                  gt_labels=None,
        #                                                  gt_bboxes_ignore=None,
        #                                                  proposal_cfg=None,
        #                                                  **kwargs)
        #     else:
        #         rpn_losses = dict(loss_rpn_cls=torch.tensor(0.).to(img.device), loss_rpn_bbox=torch.tensor(0.).to(img.device))
        # elif self.box_reg == 'vaw':
        #     vaw_seen_mask = dataset_type == 1
        #     if torch.any(vaw_seen_mask):  # 存在coco seen 目标
        #
        #         img_rpn = [x[vaw_seen_mask, ...] for x in img_f_maps]
        #         boxes_rpn = [x for idx, x in enumerate(gt_bboxes) if vaw_seen_mask[idx]]
        #         img_metas_rpn = [x for idx, x in enumerate(img_metas) if vaw_seen_mask[idx]]
        #         rpn_losses = self.rpn_head.forward_train(img_rpn,
        #                                                  img_metas_rpn,
        #                                                  boxes_rpn,
        #                                                  gt_labels=None,
        #                                                  gt_bboxes_ignore=None,
        #                                                  proposal_cfg=None,
        #                                                  **kwargs)
        #     else:
        #         rpn_losses = dict(loss_rpn_cls=torch.tensor(0.).to(img.device), loss_rpn_bbox=torch.tensor(0.).to(img.device))
        # elif self.box_reg == 'all':
        #     rpn_losses = self.rpn_head.forward_train(img_f_maps,
        #                                              img_metas,
        #                                              gt_bboxes,
        #                                              gt_labels=None,
        #                                              gt_bboxes_ignore=None,
        #                                              proposal_cfg=None,
        #                                              **kwargs)
        # else:
        #     raise NameError
        # losses.update(rpn_losses)

        # # for all proposals
        # patch_dataset_type = [dataset_type[idx] for idx, x in enumerate(gt_bboxes) for _ in range(len(x))]
        # patch_dataset_type = torch.tensor(patch_dataset_type).to(img.device)

        num_phase_per_img = [len(x) for x in phases]
        phases = [t for x in phases for t in x]
        phase_context, phase_eot_index, _ = self.prompt_catephrase_learner(phases, device=img.device)
        phase_emb = self.text_encoder(phase_context, phase_eot_index)

        patch_gt_label = torch.cat(gt_labels, dim=0)  # [label for x_per_img in gt_labels for label in x_per_img]

        # patch_gt_label = torch.stack(patch_gt_label, dim=0)
        boxes_feats, bbox_feat_maps = self.att_head(x=img_f_maps, proposal_list=gt_bboxes, cate_emb=phase_emb)

        if self.text_emb_from is None:
            prompt_context, eot_index, att_group_member_num = self.prompt_att_learner()  # 620x77x512
            text_features_att = self.text_encoder(prompt_context, eot_index)
            text_features = text_features_att
        else:
            text_features = self.text_features.to(img.device)

        # if hasattr(self, 'prompt_category_learner'):
        #     prompt_context, eot_index, cate_group_member_num = self.prompt_category_learner()  # 620x77x512
        #     text_features_cate = self.text_encoder(prompt_context, eot_index)
        #     text_features.append(text_features_cate)
        # text_features = torch.cat(text_features, dim=0)

        boxes_feats = boxes_feats / boxes_feats.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # extra_info = {'boxes_feats': boxes_feats}
        # if "img_crops" in kwargs and self.with_kd_model:
        #     img_crops = kwargs.get('img_crops', None)
        #     img_crops = torch.cat(img_crops, dim=0)
        #     with torch.no_grad():
        #         img_crop_features, _, _ = self.kd_model(img_crops)
        #     img_crop_features = self.kd_img_align(img_crop_features)
        #     img_crop_features = img_crop_features / img_crop_features.norm(dim=-1, keepdim=True)
        #     extra_info['img_crop_features'] = img_crop_features
        #     kd_logit_scale = self.kd_logit_scale.exp()
        #     kd_logits = kd_logit_scale * img_crop_features @ text_features.t()  # 2x(620+80)
        #     extra_info['kd_logits'] = kd_logits

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * boxes_feats @ text_features.t()  # 2x620
        att_losses = self.head.forward_lsa(logits, patch_gt_label)

        losses.update(att_losses)

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

    def test_box_free(self, img, img_metas, rescale=False, **kwargs):
        if self.with_clip_img_backbone:
            image_features, final_map, img_f_maps = self.img_backbone(img)  # 2x1024
        else:
            img_f_maps = self.img_backbone(img)
        img_f_maps = self.img_neck(img_f_maps)
        proposal_list = self.rpn_head.simple_test_rpn(img_f_maps, img_metas, with_nms=True)

        num_boxes_per_img = [len(x) for x in proposal_list]
        boxes_feats, bbox_feat_maps = self.att_head(img_f_maps, proposal_list)
        text_features = []
        if hasattr(self, 'prompt_att_learner'):
            prompt_context, eot_index, att_group_member_num = self.prompt_att_learner()  # 620x77x512
            text_features_att = self.text_encoder(prompt_context, eot_index)
            text_features.append(text_features_att)

        if hasattr(self, 'prompt_category_learner'):
            prompt_context, eot_index, cate_group_member_num = self.prompt_category_learner()  # 620x77x512
            text_features_cate = self.text_encoder(prompt_context, eot_index)
            text_features.append(text_features_cate)
        text_features = torch.cat(text_features, dim=0)

        boxes_feats = boxes_feats / boxes_feats.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * boxes_feats @ text_features.t()  # 2x620
        if hasattr(self, 'prompt_att_learner'):
            att_logit, cate_logit = logits[:, :len(text_features_att)], logits[:, len(text_features_att):]
            split_att_group_logits = att_logit.split(att_group_member_num, dim=-1)
            att_logit = [torch.max(x, dim=-1, keepdim=True)[0] for x in split_att_group_logits]
            att_logit = torch.cat(att_logit, dim=-1)
            logits = torch.cat((att_logit, cate_logit), dim=-1)

        logits = torch.split(logits, num_boxes_per_img, dim=0)

        pred_att_list = [x.detach().cpu() for x in logits]
        proposal_list = [x.detach().cpu() for x in proposal_list]
        # (tl_x, tl_y, br_x, br_y, score)
        # proposal_list

        if rescale:
            for proposals, meta in zip(proposal_list, img_metas):
                proposals[:, :4] /= proposals.new_tensor(meta['scale_factor'])

        results = []
        for proposals, pred_att in zip(proposal_list, pred_att_list):
            results.append(torch.cat((proposals, pred_att), dim=1))

        return results

    def test_box_given(self, img, gt_bboxes, phases):
        if self.with_clip_img_backbone:
            image_features, final_map, img_f_maps = self.img_backbone(img)  # 2x1024
        else:
            img_f_maps = self.img_backbone(img)
        img_f_maps = self.img_neck(img_f_maps)

        phases = [t for x in phases for t in x]
        phase_context, phase_eot_index, _ = self.prompt_catephrase_learner(phases, device=img.device)
        phase_emb = self.text_encoder(phase_context, phase_eot_index)

        num_boxes_per_img = [len(x) for x in gt_bboxes]
        boxes_feats, bbox_feat_maps = self.att_head(x=img_f_maps, proposal_list=gt_bboxes, cate_emb=phase_emb)

        if self.text_emb_from is None:
            prompt_context, eot_index, att_group_member_num = self.prompt_att_learner()  # 620x77x512
            text_features_att = self.text_encoder(prompt_context, eot_index)
            text_features = text_features_att
        else:
            text_features = self.text_features.to(img.device)

        boxes_feats = boxes_feats / boxes_feats.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * boxes_feats @ text_features.t()  # 2x620
        logits = torch.split(logits, num_boxes_per_img, dim=0)
        pred = [x.detach().cpu().numpy() for x in logits]
        return pred

    def simple_test(self, img, img_metas, phases, gt_bboxes=None, rescale=False, **kwargs):
        if self.test_content == 'box_given':
            assert gt_bboxes is not None
            gt_bboxes = gt_bboxes[0]
            return self.test_box_given(img, gt_bboxes, phases[0])
        elif self.test_content == 'box_free':
            return self.test_box_free(img, img_metas, phases, rescale=rescale, **kwargs)
        else:
            raise NotImplementedError
