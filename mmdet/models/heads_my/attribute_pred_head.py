import warnings
from abc import abstractmethod
import torch
import torch.nn as nn
import numpy as np
from mmcv.runner import force_fp32

from ..builder import HEADS, build_loss
from mmcv.runner import BaseModule
from mmdet.datasets_my.evaluate_tools import cal_metrics


@HEADS.register_module()
class AttributePredHead(BaseModule):
    def __init__(self,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None
                 ):
        super(AttributePredHead, self).__init__(init_cfg)
        self.loss_cls = build_loss(loss_cls)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def loss(self,
             cls_logits,
             gt_labels
             ):
        loss = self.loss_cls(cls_logits, gt_labels)
        pred_prob = cls_logits.sigmoid()

        pred_att = pred_prob > 0.5
        gt_labels = gt_labels == 1

        t_p_samples = torch.sum(pred_att * gt_labels).float()
        gt_p_samples = torch.sum(gt_labels).float()
        pred_p_samples = torch.sum(pred_att).float()

        losses = {
            "loss": loss,
            "recall": t_p_samples / gt_p_samples,
            'precision': t_p_samples / pred_p_samples,
            't_p_samples': t_p_samples,
            'pred_positive_sample': pred_p_samples

        }
        return losses

    def forward_train(self,
                      proposal_flatten_features,
                      proposal_attribute_features,
                      proposal_att_list,
                      unique_attribute_idxs,
                      **kwargs):
        # normalized features
        roi_features = proposal_flatten_features / proposal_flatten_features.norm(dim=-1, keepdim=True)
        att_features = proposal_attribute_features / proposal_attribute_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * roi_features @ att_features.t()
        logits_per_text = logit_scale * att_features @ roi_features.t()
        each_proposal_atts_list = [atts for x in proposal_att_list for atts in x]
        gt_label = torch.zeros((len(roi_features), len(att_features))).to(roi_features.device)
        for proposal_idx, each_proposal_atts in enumerate(each_proposal_atts_list):
            mask = unique_attribute_idxs.view(1, -1) - each_proposal_atts.view(-1, 1)
            mask = torch.sum(mask == 0, dim=0)
            gt_label[proposal_idx] = mask
        # import pdb
        # pdb.set_trace()
        loss = self.loss(logits_per_image, gt_label)
        # shape = [global_batch_size, global_batch_size]
        return loss, logits_per_image, logits_per_text

    def simple_test(self, proposal_flatten_features, proposal_attribute_features):
        roi_features = proposal_flatten_features / proposal_flatten_features.norm(dim=-1, keepdim=True)
        att_features = proposal_attribute_features / proposal_attribute_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * roi_features @ att_features.t()  # 2x620

        pred = list(logits.detach().cpu().numpy())

        return pred


    def forward(self, feats):
        return feats

