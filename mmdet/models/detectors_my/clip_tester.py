import json

import torch
from mmcv.runner import get_dist_info
from torch import nn
from ..backbones_my.clip import _MODELS, _download, build_model, tokenize
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from ..detectors.base import BaseDetector
import warnings


# 使用CLIP测试
@DETECTORS.register_module()
class CLIP_Tester(BaseDetector):
    def __init__(self,
                 attribute_index_file,
                 backbone_name,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(CLIP_Tester, self).__init__(init_cfg)

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

        url = _MODELS[backbone_name]
        load_ckpt_from = _download(url)
        model = torch.jit.load(load_ckpt_from, map_location="cpu").eval()
        new_dict = model.state_dict()
        # import pdb
        # pdb.set_trace()
        self.model = build_model(new_dict)

    def extract_feat(self, img):
        return img

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

    def forward_train(
            self,
            img,
            img_metas,
            gt_labels,
            gt_bboxes_ignore=None,
            data_set_type=None,
            **kwargs
    ):
        image_features, last_f_map, f_maps = self.image_encoder(img)  # 2x1024
        # prompts = self.prompt_learner()  # 620x77x512
        # tokenized_prompts = self.tokenized_prompts
        # text_features = self.text_encoder(prompts, tokenized_prompts)  # 620x1024
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
        # import pdb
        # pdb.set_trace()
        if hasattr(self, 'img_proj_head'):
            image_features = getattr(self, 'img_proj_head')(image_features)
        if hasattr(self, 'text_proj_head'):
            text_features = getattr(self, 'text_proj_head')(text_features)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()  # 2x620

        if hasattr(self, 'prompt_att_learner'):
            att_logit, cate_logit = logits[:, :len(text_features_att)], logits[:, len(text_features_att):]
            split_att_group_logits = att_logit.split(att_group_member_num, dim=-1)
            att_logit = [torch.mean(x, dim=-1, keepdim=True) for x in split_att_group_logits]
            att_logit = torch.cat(att_logit, dim=-1)
            logits = torch.cat((att_logit, cate_logit), dim=-1)

        losses = self.bbox_head.forward_train(logits, img_metas, data_set_type, gt_labels)

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
        texts = list(self.att2id.keys())
        texts = [f'The attribute of the object is {x}' for x in texts]
        texts = tokenize(texts).to(img.device)

        image_features, _, _ = self.model.encode_image(img)
        text_features = self.model.encode_text(texts)
        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()
        probs = logits_per_image.cpu().numpy()
        # shape = [global_batch_size, global_batch_size]
        return probs

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass

