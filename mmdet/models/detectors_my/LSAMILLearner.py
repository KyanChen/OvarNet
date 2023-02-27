import json

import torch
from mmcv.runner import get_dist_info
from torch import nn, distributed
import torch.nn.functional as F

from ..builder import DETECTORS, build_backbone, build_head, build_neck, build_loss
from ..detectors.base import BaseDetector
import warnings


@DETECTORS.register_module()
class LSALearner(BaseDetector):
    def __init__(self,
                 attribute_index_file,
                 need_train_names,
                 backbone,
                 text_emb_from=None,
                 prompt_att_learner=None,
                 prompt_category_learner=None,
                 prompt_phase_learner=None,
                 prompt_caption_learner=None,
                 img_encoder=None,
                 shared_prompt_vectors=False,
                 load_prompt_weights='',
                 max_sample_att=None,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 gather_gpus=True,
                 mil_loss=None,
                 init_cfg=None):
        super(LSALearner, self).__init__(init_cfg)
        self.max_sample_att = max_sample_att
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

        clip_model = build_backbone(backbone).model
        if img_encoder is None:
            self.image_encoder = clip_model.visual
        else:
            self.image_encoder = build_backbone(img_encoder)
            self.img_proj_head = nn.Linear(768, 1024)
        self.logit_scale = nn.Parameter(clip_model.logit_scale.data)

        self.text_emb_from = text_emb_from
        if text_emb_from is None:
            print('load text emb from prompt learner')
            self.text_encoder = build_backbone(
                dict(
                    type='TextEncoder',
                    clip_model=clip_model
                )
            )

            if prompt_att_learner is not None:
                assert len(self.att2id)
                prompt_att_learner.update(
                    dict(attribute_list=list(self.att2id.keys()),
                         clip_model=clip_model,
                         self_name='prompt_att_learner'
                         )
                )
                self.prompt_att_learner = build_backbone(prompt_att_learner)
        else:
            print(f'load text emb from {text_emb_from}')
            self.text_features = torch.load(text_emb_from).cpu()

        if prompt_category_learner is not None:
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

        if prompt_phase_learner is not None:
            prompt_phase_learner.update(
                dict(clip_model=clip_model)
            )
            self.prompt_phase_learner = build_backbone(prompt_phase_learner)

        if prompt_caption_learner is not None:
            prompt_caption_learner.update(
                dict(clip_model=clip_model)
            )
            self.prompt_caption_learner = build_backbone(prompt_caption_learner)

        # if load_prompt_weights:
        #     state_dict = torch.load(prompt_learner_weights, map_location="cpu")
        #     self.prompt_learner.load_state_dict(state_dict)

        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        if neck is not None:
            self.neck = build_neck(neck)
        if bbox_head is not None:
            bbox_head.update(train_cfg=train_cfg)
            bbox_head.update(test_cfg=test_cfg)

            bbox_head['attribute_index_file'] = attribute_index_file
            self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.mil_loss = build_loss(mil_loss)
        self.gather_gpus = gather_gpus
        self.need_train_names = need_train_names

        self.rank, self.world_size = get_dist_info()

        for name, param in self.named_parameters():
            flag = False
            for need_train_name in self.need_train_names:
                if need_train_name in name:
                    flag = True
            param.requires_grad_(flag)

    def extract_feat(self, img):
        return img

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

    def val_step(self, data, optimizer=None):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def gather_features(self, features, keep_loss=True):
        batch_size = torch.tensor(features.shape[0], device=features.device)
        batch_size_full = [torch.zeros_like(batch_size) for _ in range(self.world_size)]
        distributed.all_gather(batch_size_full, batch_size)
        # cutting all data to min batch size across all GPUs
        min_bs = min([bs.item() for bs in batch_size_full])
        if min_bs < batch_size:
            features = features[:min_bs]

        gathered_features = [torch.zeros_like(features) for _ in range(self.world_size)]
        distributed.all_gather(gathered_features, features)

        if keep_loss:
            gathered_features[self.rank] = features
        gathered_features = torch.cat(gathered_features, dim=0)
        return gathered_features, min_bs

    def forward_train(
            self,
            img,
            img_metas,
            gt_bboxes_ignore=None,
            data_set_type=None,
            gt_labels=None,
            **kwargs
    ):
        img_all_feats, last_f_map, f_maps = self.image_encoder(img)
        if self.gather_gpus and self.world_size > 1:
            img_all_feats, min_bs = self.gather_features(img_all_feats)
            gt_labels, min_bs = self.gather_features(gt_labels)

        if self.max_sample_att is None:
            val_inds = None
        else:
            mask = gt_labels == 1  # NxC
            val_inds = torch.any(mask, dim=0).nonzero()[:, 0]
            if len(val_inds) > self.max_sample_att:
                val_inds = val_inds[:self.max_sample_att]
            elif len(val_inds) < self.max_sample_att:
                extra_sample_num = self.max_sample_att - len(val_inds)
                extra_inds = torch.randint(0, gt_labels.size(-1), size=[extra_sample_num]).to(img.device)
                val_inds = torch.cat((val_inds, extra_inds))
            gt_labels = gt_labels[:, val_inds]

        if self.text_emb_from is None:
            att_prompt_context, att_eot_index, att_group_member_num = self.prompt_att_learner(val_inds)  # 620x77x512
            text_all_features = self.text_encoder(att_prompt_context, att_eot_index)
        else:
            text_all_features = self.text_features.to(img.device)

        img_all_feats = img_all_feats / img_all_feats.norm(dim=-1, keepdim=True)
        text_all_features = text_all_features / text_all_features.norm(dim=-1, keepdim=True)

        # MIL Loss
        logit_scale = self.logit_scale.exp()
        img_att_scores = img_all_feats @ text_all_features.t()  # [#regions, img_batch * n_ctx]
        img_att_scores = img_att_scores * logit_scale

        # loss_mil = self.mil_loss(img_att_scores, gt_labels, weights=None, avg_positives=False)

        # pos_mask = gt_labels == 1
        # neg_mask = gt_labels == 0
        # gt_labels = gt_labels.float()
        # loss_pos = F.binary_cross_entropy_with_logits(
        #     img_att_scores[pos_mask], gt_labels[pos_mask], reduction='mean')
        # loss_neg = F.binary_cross_entropy_with_logits(
        #     img_att_scores[neg_mask], gt_labels[neg_mask], reduction='mean')
        # loss_pos_neg = loss_pos + 0.1 * loss_neg
        losses = self.bbox_head.forward_lsa(img_att_scores, gt_labels)
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

    def simple_test(self, img, img_metas, rescale=False):
        image_features, last_f_map, f_maps = self.image_encoder(img)  # 2x1024

        if self.text_emb_from is None:
            prompt_context, eot_index, att_group_member_num = self.prompt_att_learner()  # 620x77x512
            text_features_att = self.text_encoder(prompt_context, eot_index)
        else:
            text_features_att = self.text_features.to(img.device)

        # torch.save(text_features_att.cpu(), 'text_features_att_common2common_common1_5508x512_vit16.pth')
        # import pdb
        # pdb.set_trace()

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features_att / text_features_att.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        # logit_scale = 1e-1
        logits = logit_scale * image_features @ text_features.t()  # 2x620
        pred = list(logits.detach().cpu().numpy())
        return pred


    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass

