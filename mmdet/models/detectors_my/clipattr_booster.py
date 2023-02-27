import json

import torch
from mmcv.runner import get_dist_info
from torch import nn, distributed
import torch.nn.functional as F

from ..builder import DETECTORS, build_backbone, build_head, build_neck, build_loss
from ..detectors.base import BaseDetector
import warnings


@DETECTORS.register_module()
class CLIPAttr_Booster(BaseDetector):
    def __init__(self,
                 attribute_index_file,
                 need_train_names,
                 backbone,
                 prompt_att_learner=None,
                 prompt_category_learner=None,
                 prompt_phase_learner=None,
                 prompt_caption_learner=None,
                 img_encoder=None,
                 shared_prompt_vectors=False,
                 load_prompt_weights='',
                 matching_temp=0.1,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 gather_gpus=True,
                 mil_loss=None,
                 init_cfg=None):
        super(CLIPAttr_Booster, self).__init__(init_cfg)
        # self.matching_temp = matching_temp
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
        # 'img', 'biggestproposal',
        # 'img_crops', 'crops_logits',
        # 'crops_labels', 'caption', 'phases'
        # biggestproposal = kwargs['biggestproposal']
        img_crops = torch.cat(kwargs['img_crops'], dim=0)
        crops_logits = torch.cat(kwargs['crops_logits'], dim=0)
        crops_labels = torch.cat(kwargs['crops_labels'], dim=0)
        caption = kwargs['caption']
        phases = kwargs['phases']

        # region features
        img_all = torch.cat((img, img_crops), dim=0)
        img_all_feats, last_f_map, f_maps = self.image_encoder(img_all)

        # attributes all
        att_prompt_context, att_eot_index, att_group_member_num = self.prompt_att_learner()  # 620x77x512
        cate_prompt_context, cate_eot_index, cate_group_member_num = self.prompt_category_learner()  # 620x77x512

        num_phase_per_img = [len(x) for x in phases]
        phases = [t for x in phases for t in x]
        phase_context, phase_eot_index, _ = self.prompt_phase_learner(phases, device=img.device)

        caption_context, caption_eot_index, _ = self.prompt_caption_learner(caption, device=img.device)

        all_prompt_context = torch.cat([att_prompt_context, cate_prompt_context, phase_context, caption_context], dim=0)
        all_eot_index = torch.cat([att_eot_index, cate_eot_index, phase_eot_index, caption_eot_index], dim=0)
        text_all_features = self.text_encoder(all_prompt_context, all_eot_index)

        logit_scale = self.logit_scale.exp()
        img_all_feats = img_all_feats / img_all_feats.norm(dim=-1, keepdim=True)
        text_all_features = text_all_features / text_all_features.norm(dim=-1, keepdim=True)

        # caption-img, caption-biggest_proposal
        img_b_feats = img_all_feats[:len(img)]
        text_cap_feats = text_all_features[-len(caption):]

        if self.gather_gpus and self.world_size > 1:
            img_b_feats, min_bs = self.gather_features(img_b_feats)
            text_cap_feats, min_bs = self.gather_features(text_cap_feats)

        losses = {}

        # NCE biggest proposal - caption
        img_cap_scores = img_b_feats @ text_cap_feats.t()  # [#regions, img_batch * n_ctx]
        img_cap_scores = img_cap_scores * logit_scale
        img_cap_contrast_target = torch.arange(len(img_cap_scores)).to(img_cap_scores.device)
        cap_row_loss = F.cross_entropy(img_cap_scores, img_cap_contrast_target)
        cap_col_loss = F.cross_entropy(img_cap_scores.t(), img_cap_contrast_target)
        losses["loss_bp_cap_nce"] = (cap_row_loss + cap_col_loss) / 2.0

        # NCE biggest proposal - phase
        phase_start = len(att_prompt_context) + len(cate_prompt_context)
        phase_end = phase_start + len(phases)
        allpha_feats = text_all_features[phase_start: phase_end]
        allpha_feats = torch.split(allpha_feats, num_phase_per_img, dim=0)
        samp_pha_feats = [x[torch.randint(0, len(x), size=[1])] for x in allpha_feats if len(x) > 0]
        selected_phase_embs = torch.cat(samp_pha_feats, dim=0)
        mask_has_phase = torch.tensor(num_phase_per_img, device=img.device) > 0
        bg_img_feats = img_all_feats[:len(img)][mask_has_phase]
        if self.gather_gpus and self.world_size > 1:
            selected_phase_embs, min_bs = self.gather_features(selected_phase_embs)
            bg_img_feats, min_bs = self.gather_features(bg_img_feats)
        img_pha_scores = bg_img_feats @ selected_phase_embs.t()
        img_pha_scores = img_pha_scores * logit_scale
        img_pha_contrast_target = torch.arange(len(img_pha_scores)).to(img_pha_scores.device)
        pha_row_loss = F.cross_entropy(img_pha_scores, img_pha_contrast_target)
        pha_col_loss = F.cross_entropy(img_pha_scores.t(), img_pha_contrast_target)
        losses["loss_bp_pha_nce"] = (pha_row_loss + pha_col_loss) / 2.0

        # MIL biggest proposal - phase
        phase_start = len(att_prompt_context) + len(cate_prompt_context)
        phase_end = phase_start + len(phases)
        img_allpha_scores = img_all_feats[:len(img)] @ text_all_features[phase_start: phase_end].t()
        img_allpha_scores = img_allpha_scores * logit_scale
        allpha_labels = torch.zeros_like(img_allpha_scores, device=img.device)
        flag_set_cursor = 0
        for idx_sample, num_pha in enumerate(num_phase_per_img):
            allpha_labels[idx_sample, flag_set_cursor:flag_set_cursor + num_pha] = 1
            flag_set_cursor += num_pha
        loss_bp_pha_mil = self.mil_loss(img_allpha_scores, allpha_labels, weights=None, avg_positives=False)
        losses["loss_bp_pha_mil"] = loss_bp_pha_mil

        # MIL crops - att
        crop_att_scores = img_all_feats[len(img):] @ text_all_features[:len(att_prompt_context)].t()
        crop_att_scores = crop_att_scores * logit_scale
        loss_crop_att_mil = self.mil_loss(crop_att_scores, crops_labels[:, :len(self.att2id)], weighted_unk=5.)
        losses["loss_crop_att_mil"] = loss_crop_att_mil

        # MIL crops - cate
        cate_start = len(att_prompt_context)
        cate_end = cate_start + len(cate_prompt_context)
        crop_cate_scores = img_all_feats[len(img):] @ text_all_features[cate_start: cate_end].t()
        crop_cate_scores = crop_cate_scores * logit_scale
        loss_crop_cate_mil = self.mil_loss(crop_cate_scores, crops_labels[:, len(self.att2id):], weights=None,
                                           avg_positives=False)
        losses["loss_crop_cate_mil"] = loss_crop_cate_mil

        # MIL biggest proposal - att
        img_att_scores = img_all_feats[:len(img)] @ text_all_features[: len(att_prompt_context)].t()
        img_att_scores = img_att_scores * logit_scale
        loss_bp_att_mil = self.mil_loss(img_att_scores, gt_labels[:, :len(self.att2id)], weighted_unk=10.)
        losses["loss_bp_att_mil"] = loss_bp_att_mil

        # MIL biggest proposal - cate
        cate_start = len(att_prompt_context)
        cate_end = cate_start + len(cate_prompt_context)
        img_cate_scores = img_all_feats[:len(img)] @ text_all_features[cate_start: cate_end].t()
        img_cate_scores = img_cate_scores * logit_scale
        loss_bp_cate_mil = self.mil_loss(img_cate_scores, gt_labels[:, len(self.att2id):], weights=None, avg_positives=False)
        losses["loss_bp_cate_mil"] = loss_bp_cate_mil

        # KLLoss crops - cate_att
        kl_att_loss = F.kl_div(F.log_softmax(crop_att_scores, dim=-1), F.softmax(crops_logits[:, :len(self.att2id)]),
                               reduction='batchmean')  # input is log-probabilities, target is probabilities
        kl_cate_loss = F.kl_div(F.log_softmax(crop_cate_scores, dim=-1), F.softmax(crops_logits[:, len(self.att2id):]),
                                reduction='batchmean')  # input is log-probabilities, target is probabilities
        losses["loss_kl_att"] = kl_att_loss
        losses["loss_kl_cate"] = kl_cate_loss
        # print(losses)
        # import pdb
        # pdb.set_trace()
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

        # prompt_context = self.prompt_learner()  # 620x77x512
        # text_features = self.text_encoder(prompt_context, self.tokenized_prompts)

        if hasattr(self, 'img_proj_head'):
            image_features = getattr(self, 'img_proj_head')(image_features)
        if hasattr(self, 'text_proj_head'):
            text_features = getattr(self, 'text_proj_head')(text_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # logit_scale = self.logit_scale.exp()
        logit_scale = 1e-1
        logits = logit_scale * image_features @ text_features.t()  # 2x620
        if hasattr(self, 'prompt_att_learner'):
            att_logit, cate_logit = logits[:, :len(text_features_att)], logits[:, len(text_features_att):]
            split_att_group_logits = att_logit.split(att_group_member_num, dim=-1)
            att_logit = [torch.max(x, dim=-1, keepdim=True)[0] for x in split_att_group_logits]
            att_logit = torch.cat(att_logit, dim=-1)
            logits = torch.cat((att_logit, cate_logit), dim=-1)

        pred = list(logits.detach().cpu().numpy())
        return pred


    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass

