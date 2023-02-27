import json
import os
import warnings
from abc import abstractmethod

import numpy as np
import torch
from einops import rearrange, repeat
from mmcv.runner import force_fp32

from ..builder import HEADS, build_loss
from mmcv.runner import BaseModule
from mmdet.datasets_my.evaluate_tools import cal_metrics
import torch.nn.functional as F
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm
import warnings
from torchmetrics.functional import precision_recall, f1_score, average_precision

warnings.filterwarnings('ignore')

@HEADS.register_module()
class PromptHead(BaseModule):
    def __init__(self,
                 attr_freq_file=None,
                 category_freq_file=None,
                 attribute_index_file=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 re_weight_different_att=0.2,  # 0.2:68, 0.4:67
                 re_weight_category=1,
                 re_weight_gamma=2,
                 re_weight_beta=0.995,  # 越小，加权越弱
                 balance_unk=0.1,
                 kd_model_loss=None,
                 balance_kd=0.1,
                 balance_capdata=0.5,
                 balance_teacher_loss=0.5
                 ):
        super(PromptHead, self).__init__(init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.attribute_index_file = attribute_index_file
        self.att2id = {}
        self.att_seen_unseen = {}
        if 'att_file' in attribute_index_file.keys():
            file = attribute_index_file['att_file']
            att2id = json.load(open(file, 'r'))
            att_group = attribute_index_file['att_group']
            if att_group in ['common1', 'common2']:
                self.att2id = att2id[att_group]
                self.att_seen_unseen['seen'] = list(att2id['common1'].keys())
                self.att_seen_unseen['unseen'] = list(att2id['common2'].keys())
            elif att_group in ['common', 'rare']:
                self.att2id = att2id[att_group]
                self.att_seen_unseen['seen'] = list(att2id['common'].keys())
                self.att_seen_unseen['unseen'] = list(att2id['rare'].keys())
            elif att_group in ['base', 'novel']:
                self.att2id = att2id[att_group]
                self.att_seen_unseen['seen'] = list(att2id['base'].keys())
                self.att_seen_unseen['unseen'] = list(att2id['novel'].keys())
            elif att_group == 'common1+common2':
                self.att2id.update(att2id['common1'])
                self.att2id.update(att2id['common2'])
                self.att_seen_unseen['seen'] = list(att2id['common1'].keys())
                self.att_seen_unseen['unseen'] = list(att2id['common2'].keys())
            elif att_group == 'common+rare':
                self.att2id.update(att2id['common'])
                self.att2id.update(att2id['rare'])
                self.att_seen_unseen['seen'] = list(att2id['common'].keys())
                self.att_seen_unseen['unseen'] = list(att2id['rare'].keys())
            elif att_group == 'base+novel':
                self.att2id.update(att2id['base'])
                self.att2id.update(att2id['novel'])
                self.att_seen_unseen['seen'] = list(att2id['base'].keys())
                self.att_seen_unseen['unseen'] = list(att2id['novel'].keys())
            else:
                raise NameError
        self.category2id = {}
        self.category_seen_unseen = {}
        if 'category_file' in attribute_index_file.keys():
            file = attribute_index_file['category_file']
            category2id = json.load(open(file, 'r'))
            att_group = attribute_index_file['category_group']
            if att_group in ['common1', 'common2', 'common', 'rare']:
                self.category2id = category2id[att_group]
                self.category_seen_unseen['seen'] = list(category2id['common1'].keys())
                self.category_seen_unseen['unseen'] = list(category2id['common2'].keys())
            elif att_group == 'common1+common2':
                self.category2id.update(category2id['common1'])
                self.category2id.update(category2id['common2'])
                self.category_seen_unseen['seen'] = list(category2id['common1'].keys())
                self.category_seen_unseen['unseen'] = list(category2id['common2'].keys())
            elif att_group == 'common+rare':
                self.category2id.update(category2id['common'])
                self.category2id.update(category2id['rare'])
            else:
                raise NameError
        self.att2id = {k: v - min(self.att2id.values()) for k, v in self.att2id.items()}
        self.category2id = {k: v - min(self.category2id.values()) for k, v in self.category2id.items()}

        self.re_weight_different_att = re_weight_different_att

        if attr_freq_file is not None and len(self.att2id):
            attr_freq = json.load(open(attr_freq_file, 'r'))
            self.reweight_att_frac = self.reweight_att(attr_freq, self.att2id)
        if category_freq_file is not None and len(self.category2id):
            category_freq = json.load(open(category_freq_file, 'r'))
            self.reweight_cate_frac = self.reweight_att(category_freq, self.category2id)

        self.re_weight_gamma = re_weight_gamma
        self.re_weight_beta = re_weight_beta

        self.re_weight_category = re_weight_category

        self.balance_unk = balance_unk
        self.kd_model_loss = kd_model_loss
        self.balance_kd = balance_kd
        self.balance_teacher_loss = balance_teacher_loss
        self.balance_capdata = balance_capdata

    def reweight_att(self, attr_freq, att2id):
        refine_attr_freq = {}
        idx_pre = -1
        for att, idx in att2id.items():
            assert idx > idx_pre
            idx_pre = idx
            refine_attr_freq[att] = attr_freq[att]

        # pos_rew = torch.from_numpy(np.array([v['pos'] for k, v in refine_attr_freq.items()], dtype=np.float32))
        # neg_rew = torch.from_numpy(np.array([v['neg'] for k, v in refine_attr_freq.items()], dtype=np.float32))
        total_rew_bak = torch.from_numpy(np.array([v['total'] for k, v in refine_attr_freq.items()], dtype=np.float32))

        # total_rew = 99 * (total_rew_bak - total_rew_bak.min()) / (total_rew_bak.max() - total_rew_bak.min()) + 1
        # total_rew = 1 - torch.pow(self.re_weight_beta, total_rew)
        # total_rew = (1 - self.re_weight_beta) / total_rew
        # total_rew = 620 * total_rew / total_rew.sum()

        total_rew = 1 / torch.pow(total_rew_bak, self.re_weight_different_att)
        total_rew = len(refine_attr_freq) * total_rew / total_rew.sum()
        # import pdb
        # pdb.set_trace()
        return total_rew

    def get_classify_loss(self, cls_scores, gt_labels, balance_unk=1., reweight=None):
        # cls_scores: BxN
        # gt_labels: BxN
        BS = cls_scores.size(0)
        cls_scores_flatten = rearrange(cls_scores, 'B N -> (B N)')
        gt_labels_flatten = rearrange(gt_labels, 'B N -> (B N)')
        gt_labels_flatten = gt_labels_flatten.float()
        if reweight is not None:
            total_rew = repeat(reweight, 'N -> (B N)', B=BS)
        pos_mask = gt_labels_flatten == 1
        neg_mask = gt_labels_flatten == 0
        unk_mask = gt_labels_flatten == 2

        # cls_scores_flatten = torch.sigmoid(cls_scores_flatten)
        # pos_pred = torch.clamp(cls_scores_flatten[pos_mask], 1e-10, 1-1e-10)
        # neg_pred = torch.clamp(1-cls_scores_flatten[neg_mask], 1e-10, 1-1e-10)
        # loss_pos = - total_rew[pos_mask] * torch.pow(1-cls_scores_flatten[pos_mask], self.re_weight_gamma) * torch.log(pos_pred)
        # loss_neg = - total_rew[neg_mask] * torch.pow(cls_scores_flatten[neg_mask], self.re_weight_gamma) * torch.log(neg_pred)
        # # loss_pos = - total_rew[pos_mask] * torch.log(pos_pred)
        # # loss_neg = - total_rew[neg_mask] * torch.log(neg_pred)
        # loss_pos = loss_pos.mean()
        # loss_neg = loss_neg.mean()
        if reweight is None:
            pos_neg_rew = None
        else:
            pos_neg_rew = total_rew[~unk_mask]
        loss_pos_neg = F.binary_cross_entropy_with_logits(
            cls_scores_flatten[~unk_mask], gt_labels_flatten[~unk_mask], weight=pos_neg_rew,
            reduction='mean')

        # loss_pos = F.binary_cross_entropy_with_logits(
        #     cls_scores_flatten[pos_mask], gt_labels_flatten[pos_mask], weight=total_rew[pos_mask], reduction='mean')
        # loss_neg = F.binary_cross_entropy_with_logits(
        #     cls_scores_flatten[neg_mask], gt_labels_flatten[neg_mask], weight=total_rew[neg_mask], reduction='mean')
        # loss_pos_neg = loss_pos + 4 * loss_neg

        pred_unk = cls_scores_flatten[unk_mask]
        gt_labels_unk = pred_unk.new_zeros(pred_unk.size())

        # bce_loss_unk = F.binary_cross_entropy(pred_unk, gt_labels_unk, reduction='mean')
        # bce_loss = loss_pos + loss_neg + self.balance_unk * bce_loss_unk
        if len(pred_unk) == 0:
            bce_loss_unk = torch.tensor(0.).to(loss_pos_neg.device)
        else:
            bce_loss_unk = F.binary_cross_entropy_with_logits(pred_unk, gt_labels_unk, reduction='mean')
        bce_loss = loss_pos_neg + balance_unk * bce_loss_unk

        return bce_loss

    def loss(self,
             cls_scores,
             gt_labels,
             img_metas,
             **kwargs
             ):

        loss_s_ce = self.get_classify_loss(cls_scores, gt_labels)

        losses = {}
        losses['loss_s_ce'] = loss_s_ce

        if 'img_crop_features' in kwargs and self.kd_model_loss:
            img_crop_features = kwargs.get('img_crop_features', None)
            proposal_features = kwargs.get('boxes_feats', None)
            kd_logits = kwargs.get('kd_logits', None)

            # img_crop_sigmoid = torch.sigmoid(img_crop_features)
            # proposal_sigmoid = torch.sigmoid(proposal_features)

            # loss_kd = F.kl_div(img_crop_features, proposal_features) + F.kl_div(proposal_features, img_crop_features)
            # loss_kd = F.kl_div(proposal_features, img_crop_features, reduction='mean')

            # similarity = torch.cosine_similarity(img_crop_features, proposal_features, dim=-1)
            # loss = 1 - similarity
            if self.kd_model_loss == 'smooth-l1':
                loss_kd = F.smooth_l1_loss(proposal_features, img_crop_features, reduction='mean')
                loss_kd = self.balance_kd * loss_kd
            elif self.kd_model_loss == 'ce':
                proposal_features = torch.sigmoid(self.balance_kd * proposal_features)
                img_crop_features = torch.sigmoid(self.balance_kd * img_crop_features)
                loss_kd = F.binary_cross_entropy(proposal_features, img_crop_features, reduction='mean')
            elif self.kd_model_loss == 't_ce+ts_ce':


                # gt_labels_flatten = gt_labels.view(-1)
                # kd_logits_flatten = kd_logits.view(-1)
                # cls_scores_flatten = cls_scores.view(-1)
                # unk_mask = gt_labels_flatten == 2

                BS = gt_labels.size()
                total_rew = self.reweight_att_frac.to(gt_labels.device)
                # total_rew = repeat(total_rew, 'N -> (B N)', B=BS)

                loss_t_ce = self.get_classify_loss(kd_logits, gt_labels)
                loss_ts_ce = F.cross_entropy(cls_scores, (kd_logits.detach()).softmax(dim=-1), weight=total_rew)

                losses['loss_t_ce'] = self.balance_kd * 0.5 * loss_t_ce
                losses['loss_ts_ce'] = self.balance_kd * loss_ts_ce
            elif self.kd_model_loss == 't_ce':
                loss_t_ce = self.get_classify_loss(kd_logits, gt_labels)
                losses['loss_t_ce'] = loss_t_ce
            else:
                raise NotImplementedError

        return losses

    def forward_ovarnet_train(self,
                      pred_logits,
                      img_metas,
                      data_set_type,
                      gt_labels,
                      **kwargs):
        losses = {}

        # for vaw and coco dataset
        cate_mask = data_set_type == 0
        att_mask = data_set_type == 1
        x = pred_logits
        pred_att_logits = x[att_mask][:, :len(self.att_seen_unseen['seen'])]
        pred_cate_logits = x[cate_mask][:, len(self.att2id):len(self.att2id) + len(self.category_seen_unseen['seen'])]
        # pred_cate_logits = x[cate_mask][:, len(self.att2id):]
        gt_att = gt_labels[att_mask][:, :len(self.att_seen_unseen['seen'])]
        gt_cate = gt_labels[cate_mask][:, len(self.att2id):len(self.att2id) + len(self.category_seen_unseen['seen'])]
        # gt_cate = gt_labels[cate_mask][:, len(self.att2id):]
        if len(pred_att_logits):
            if hasattr(self, 'reweight_att_frac'):
                total_rew_att = self.reweight_att_frac.to(gt_labels.device)
            else:
                total_rew_att = None
            att_loss = self.get_classify_loss(
                pred_att_logits, gt_att, self.balance_unk, total_rew_att[:pred_att_logits.size(-1)])
            losses['att_bce_loss'] = att_loss
            losses.update(self.get_acc(pred_att_logits, gt_att, pattern='att'))
        else:
            losses['att_bce_loss'] = torch.tensor(0.).to(x.device)
            losses['att_map'] = torch.tensor(0.).to(x.device)

        if len(pred_cate_logits):
            if hasattr(self, 'reweight_cate_frac'):
                total_rew_cate = self.reweight_cate_frac.to(gt_labels.device)
            else:
                total_rew_cate = None
            cate_loss = self.get_classify_loss(
                pred_cate_logits, gt_cate, self.balance_unk, total_rew_cate[:pred_cate_logits.size(-1)])
            losses['cate_bce_loss'] = cate_loss * self.re_weight_category
            losses.update(self.get_acc(pred_cate_logits, gt_cate, pattern='cate'))
        else:
            losses['cate_bce_loss'] = torch.tensor(0.).to(x.device)
            losses['cate_map'] = torch.tensor(0.).to(x.device)

        # for caption dataset
        att_cate_mask = data_set_type == 2
        x = pred_logits
        pred_attcate_logits = x[att_cate_mask]
        gt_attcate = gt_labels[att_cate_mask]
        if len(pred_attcate_logits):
            if hasattr(self, 'reweight_att_frac') and hasattr(self, 'reweight_cate_frac'):
                total_rew_att = self.reweight_att_frac.to(gt_labels.device)
                total_rew_cate = self.re_weight_category * self.reweight_cate_frac.to(gt_labels.device)
                total_rew = torch.cat([total_rew_att, total_rew_cate], dim=0)
            else:
                total_rew = None
            cap_attcate_loss = self.get_classify_loss(
                pred_attcate_logits, gt_attcate, self.balance_unk, total_rew)
            losses['cap_attcate_loss'] = self.balance_capdata * cap_attcate_loss
            # losses.update(self.get_acc(pred_cate_logits, gt_cate, pattern='cate'))
        else:
            losses['cap_attcate_loss'] = torch.tensor(0.).to(x.device)

        # for knowledge distilation
        if 'kd_logits' in kwargs and self.kd_model_loss:
            img_crop_features = kwargs.get('img_crop_features', None)
            proposal_features = kwargs.get('boxes_feats', None)
            kd_logits = kwargs.get('kd_logits', None)

            # img_crop_sigmoid = torch.sigmoid(img_crop_features)
            # proposal_sigmoid = torch.sigmoid(proposal_features)

            # loss_kd = F.kl_div(img_crop_features, proposal_features) + F.kl_div(proposal_features, img_crop_features)
            # loss_kd = F.kl_div(proposal_features, img_crop_features, reduction='mean')

            # similarity = torch.cosine_similarity(img_crop_features, proposal_features, dim=-1)
            # loss = 1 - similarity
            if self.kd_model_loss == 'smooth-l1':
                loss_kd = F.smooth_l1_loss(proposal_features, img_crop_features, reduction='mean')
                loss_kd = self.balance_kd * loss_kd
            elif self.kd_model_loss == 'ce':
                proposal_features = torch.sigmoid(self.balance_kd * proposal_features)
                img_crop_features = torch.sigmoid(self.balance_kd * img_crop_features)
                loss_kd = F.binary_cross_entropy(proposal_features, img_crop_features, reduction='mean')
            elif 't_ce' in self.kd_model_loss and '+' in self.kd_model_loss:
                x = kd_logits
                pred_att_logits = x[att_mask][:, :len(self.att_seen_unseen['seen'])]
                gt_att = gt_labels[att_mask][:, :len(self.att_seen_unseen['seen'])]
                # pred_cate_logits = x[cate_mask][:, len(self.att2id):]
                # gt_cate = gt_labels[cate_mask][:, len(self.att2id):]
                pred_cate_logits = x[cate_mask][:,
                                   len(self.att2id):len(self.att2id) + len(self.category_seen_unseen['seen'])]
                gt_cate = gt_labels[cate_mask][:,
                          len(self.att2id):len(self.att2id) + len(self.category_seen_unseen['seen'])]
                if len(pred_att_logits):
                    if hasattr(self, 'reweight_att_frac'):
                        total_rew_att = self.reweight_att_frac.to(gt_labels.device)
                    else:
                        total_rew_att = None
                    att_loss = self.get_classify_loss(
                        pred_att_logits, gt_att,
                        self.balance_unk,
                        total_rew_att[:pred_att_logits.size(-1)]
                    )
                else:
                    att_loss = torch.tensor(0., device=pred_att_logits.device)

                if len(pred_cate_logits):
                    if hasattr(self, 'reweight_cate_frac'):
                        total_rew_cate = self.reweight_cate_frac.to(gt_labels.device)
                    else:
                        total_rew_cate = None
                    cate_loss = self.get_classify_loss(
                        pred_cate_logits, gt_cate,
                        self.balance_unk,
                        total_rew_cate[:pred_cate_logits.size(-1)]
                    )
                else:
                    cate_loss = torch.tensor(0., device=pred_att_logits.device)
                losses['t_ce_loss'] = (
                                                  att_loss + cate_loss * self.re_weight_category) * self.balance_teacher_loss * self.balance_kd

                # loss_ts_ce = F.cross_entropy(pred_logits, (kd_logits.detach()).softmax(dim=-1))

                # loss_ts_ce = 0.5 * F.cross_entropy(pred_logits[:, :len(self.att2id)], (kd_logits[:, :len(self.att2id)].detach()).softmax(dim=-1)) + \
                # F.cross_entropy(pred_logits[:, len(self.att2id):], (kd_logits[:, len(self.att2id):].detach()).softmax(dim=-1))
                # matching_temp = 0.01
            if 'ts_ce' in self.kd_model_loss and '+' in self.kd_model_loss:
                loss_ts_ce = 0.5 * F.kl_div(
                    F.log_softmax(pred_logits[:, :len(self.att2id)], dim=-1),
                    F.softmax(kd_logits[:, :len(self.att2id)].detach(), dim=-1), reduction='batchmean') + \
                             F.kl_div(F.log_softmax(pred_logits[:, len(self.att2id):], dim=-1),
                                      F.softmax(kd_logits[:, len(self.att2id):].detach(), dim=-1),
                                      reduction='batchmean')

                losses['t_s_ce_loss'] = self.balance_kd * 2 * loss_ts_ce

            elif 'ts_l1' in self.kd_model_loss and '+' in self.kd_model_loss:
                loss_ts_l1 = F.l1_loss(proposal_features, img_crop_features, reduction='mean')
                losses['t_s_l1_loss'] = self.balance_kd * 2 * loss_ts_l1
            elif 'ts_l2' in self.kd_model_loss and '+' in self.kd_model_loss:
                loss_ts_l2 = F.mse_loss(proposal_features, img_crop_features, reduction='mean')
                losses['t_s_l2_loss'] = self.balance_kd * 2 * loss_ts_l2
            elif self.kd_model_loss == 't_ce':
                loss_t_ce = self.get_classify_loss(kd_logits, gt_labels)
                losses['loss_t_ce'] = loss_t_ce
            else:
                raise NotImplementedError

        if 'logits_phase_cap' in kwargs:
            logits_phase_cap = kwargs.get('logits_phase_cap', None)
            label_phase_cap = kwargs.get('label_phase_cap', None)
            loss_phase_cap = F.binary_cross_entropy_with_logits(logits_phase_cap, label_phase_cap,
                                                                reduction='mean')
            losses['loss_phase_cap'] = self.balance_capdata * 0.5 * loss_phase_cap

        return losses

    def forward_ovarnetp_train(
            self,
            logits,
            labels,
            pos_flag,
            sampling_results,
            instance_sources,
            img_metas,
            **kwargs
    ):
        pos_flag = pos_flag.bool()
        losses = {}

        pos_patch_instance_sources = []
        for idx, res in enumerate(sampling_results):
            pos_patch_instance_sources += instance_sources[idx][res.pos_assigned_gt_inds]
        pos_patch_instance_sources = torch.tensor(pos_patch_instance_sources).to(logits.device)

        # supervised train

        # attr positive patch seen label
        mask = pos_patch_instance_sources == 1
        pred_logits = logits[pos_flag][mask][:, :len(self.att_seen_unseen['seen'])]
        gt_label = labels[pos_flag][mask][:, :len(self.att_seen_unseen['seen'])]
        if len(pred_logits):
            if hasattr(self, 'reweight_att_frac'):
                total_rew_att = self.reweight_att_frac.to(gt_label.device)
            else:
                total_rew_att = None
            att_loss = self.get_classify_loss(
                pred_logits, gt_label, self.balance_unk, total_rew_att[:pred_logits.size(-1)])
            losses['att_p_seen_bce_loss'] = att_loss
            losses.update(self.get_acc(pred_logits, gt_label, pattern='att'))
        else:
            losses['att_p_seen_bce_loss'] = torch.tensor(0.).to(logits.device)
            losses['att_map'] = torch.tensor(0.).to(logits.device)

        # cate positive patch seen label
        mask = pos_patch_instance_sources == 0
        # will cause all unseen disappear
        # pred_logits = logits[mask][:, len(self.att2id):]
        # gt_label = labels[mask][:, len(self.att2id):]
        pred_logits = logits[pos_flag][mask][:, len(self.att2id):len(self.att2id) + len(self.category_seen_unseen['seen'])]
        gt_label = labels[pos_flag][mask][:, len(self.att2id):len(self.att2id) + len(self.category_seen_unseen['seen'])]
        if len(pred_logits):
            if hasattr(self, 'reweight_cate_frac'):
                total_rew_cate = self.reweight_cate_frac.to(logits.device)
            else:
                total_rew_cate = None
            cate_loss = self.get_classify_loss(
                pred_logits, gt_label, self.balance_unk, total_rew_cate[:pred_logits.size(-1)])
            losses['cate_p_seen_bce_loss'] = cate_loss * self.re_weight_category
            losses.update(self.get_acc(pred_logits, gt_label, pattern='cate'))
        else:
            losses['cate_p_seen_bce_loss'] = torch.tensor(0.).to(logits.device)
            losses['cate_map'] = torch.tensor(0.).to(logits.device)

        # attr negative patch seen label
        # they are not annotated with high possibility.

        # cate negative patch seen label
        # they are not belong to the seen categories.
        cate_img_mask = [torch.ones(len(res.neg_inds) + len(res.pos_inds), device=logits.device) * torch.any((x == 0) + (x == 3)) for x, res in zip(instance_sources, sampling_results)]
        cate_img_mask = torch.cat(cate_img_mask).to(logits.device).bool()
        mask = cate_img_mask & (~pos_flag)
        pred_logits = logits[mask][:, len(self.att2id):len(self.att2id) + len(self.category_seen_unseen['seen'])]
        gt_label = torch.zeros_like(pred_logits, device=logits.device)
        if len(pred_logits):
            if hasattr(self, 'reweight_cate_frac'):
                total_rew_cate = self.reweight_cate_frac.to(logits.device)
            else:
                total_rew_cate = None
            cate_loss = self.get_classify_loss(
                pred_logits, gt_label, self.balance_unk, total_rew_cate[:pred_logits.size(-1)])
            losses['cate_n_seen_bce_loss'] = cate_loss * self.re_weight_category
        else:
            losses['cate_n_seen_bce_loss'] = torch.tensor(0.).to(logits.device)

        # unsupervised train
        # for knowledge distilation

        if 'kd_logits' in kwargs and self.kd_model_loss:
            kd_logits = kwargs.get('kd_logits', None)
            kd_logits = torch.split(kd_logits, [res.num_gts for res in sampling_results], dim=0)
            pos_kd_logits_list = [kd_logit[res.pos_assigned_gt_inds] for res, kd_logit in
                                  zip(sampling_results, kd_logits)]
            kd_logits = torch.cat(pos_kd_logits_list, 0)

            # supervised teacher
            # attr positive patch seen label
            mask = pos_patch_instance_sources == 1
            pred_logits = kd_logits[mask][:, :len(self.att_seen_unseen['seen'])]
            gt_label = labels[pos_flag][mask][:, :len(self.att_seen_unseen['seen'])]
            if len(pred_logits):
                if hasattr(self, 'reweight_att_frac'):
                    total_rew_att = self.reweight_att_frac.to(gt_label.device)
                else:
                    total_rew_att = None
                att_loss = self.get_classify_loss(
                    pred_logits, gt_label, self.balance_unk, total_rew_att[:pred_logits.size(-1)])
            else:
                att_loss = torch.tensor(0.).to(logits.device)

            # cate positive patch seen label
            mask = pos_patch_instance_sources == 0
            pred_logits = kd_logits[mask][:, len(self.att2id):len(self.att2id) + len(self.category_seen_unseen['seen'])]
            gt_label = labels[pos_flag][mask][:, len(self.att2id):len(self.att2id) + len(self.category_seen_unseen['seen'])]
            if len(pred_logits):
                if hasattr(self, 'reweight_cate_frac'):
                    total_rew_cate = self.reweight_cate_frac.to(logits.device)
                else:
                    total_rew_cate = None
                cate_loss = self.get_classify_loss(
                    pred_logits, gt_label, self.balance_unk, total_rew_cate[:pred_logits.size(-1)])
                cate_loss = cate_loss * self.re_weight_category
            else:
                cate_loss = torch.tensor(0.).to(logits.device)

            losses['t_ce_loss'] = (att_loss + cate_loss) * self.balance_teacher_loss * self.balance_kd

            # dist knowledge to student

            # attr positive patch, all seen and unseen classes
            pred_student_logits = logits[pos_flag]
            pred_teacher_logits = kd_logits
            ts_kl_loss = 0.5 * F.kl_div(F.log_softmax(pred_student_logits[:, :len(self.att2id)], dim=-1),
                                        F.softmax(pred_teacher_logits[:, :len(self.att2id)].detach(), dim=-1), reduction='batchmean') + \
                         F.kl_div(F.log_softmax(pred_student_logits[:, len(self.att2id):], dim=-1),
                                  F.softmax(pred_teacher_logits[:, len(self.att2id):].detach(), dim=-1),
                                  reduction='batchmean')

            losses['ts_kl_loss'] = self.balance_kd * 2 * ts_kl_loss

        return losses

    def get_acc(self, cls_scores, gt_labels, pattern='att'):
        acces = {}
        if pattern == 'att':
            pred_att_logits = cls_scores.detach().sigmoid()
            gt_att = gt_labels.detach()

            prs = []
            for i_att in range(pred_att_logits.shape[1]):
                y = gt_att[:, i_att]
                pred = pred_att_logits[:, i_att]
                gt_y = y[~(y == 2)]
                pred = pred[~(y == 2)]
                if len(pred) != 0:
                    pr = average_precision(pred, gt_y, pos_label=1)
                    if torch.isnan(pr):
                        continue
                    prs.append(pr)
            if len(prs):
                acces['att_map'] = torch.mean(torch.stack(prs))
            else:
                acces['att_map'] = torch.tensor(0.).to(cls_scores.device)

        elif pattern == 'cate':
            pred_logits = cls_scores.detach().sigmoid()
            # pred_cate_logits = pred_cate_logits.float().softmax(dim=-1).cpu()

            pred_cate_logits = pred_logits * (pred_logits == pred_logits.max(dim=-1)[0][:, None])
            gt_cate = gt_labels.detach()

            prs = []
            for i_att in range(pred_cate_logits.shape[1]):
                y = gt_cate[:, i_att]
                pred = pred_cate_logits[:, i_att]
                gt_y = y[~(y == 2)]
                pred = pred[~(y == 2)]
                if len(pred) != 0:
                    pr = average_precision(pred, gt_y, pos_label=1)
                    if torch.isnan(pr):
                        continue
                    prs.append(pr)
            if len(prs):
                acces['cate_map'] = torch.mean(torch.stack(prs))
            else:
                acces['cate_map'] = torch.tensor(0.).to(cls_scores.device)
        return acces

    def forward(self, feats):
        return feats

    def forward_lsa(self,
                    pred_logits,
                    gt_labels,
                    img_metas=None,
                    data_set_type=None,
                    **kwargs):
        losses = {}
        if hasattr(self, 'reweight_att_frac'):
            total_rew_att = self.reweight_att_frac.to(gt_labels.device)
        else:
            total_rew_att = None
        att_loss = self.get_classify_loss(pred_logits, gt_labels, 1, total_rew_att[:pred_logits.size(-1)])
        losses['att_bce_loss'] = 1000 * att_loss
        return losses


@HEADS.register_module()
class TransformerEncoderHead(BaseModule):
    def __init__(self,
                 in_dim=1024,
                 embed_dim=256,
                 use_abs_pos_embed=False,
                 drop_rate=0.1,
                 class_token=False,
                 num_encoder_layers=3,
                 global_pool=False,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 ):
        super(TransformerEncoderHead, self).__init__(init_cfg)
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.use_abs_pos_embed = use_abs_pos_embed
        self.class_token = class_token
        self.global_pool = global_pool

        if self.class_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        if self.use_abs_pos_embed:
            self.absolute_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, self.in_channel) * .02)

            self.drop_after_pos = nn.Dropout(p=drop_rate)

        self.proj1 = nn.Linear(in_dim, embed_dim)
        self.transformer_decoder = self.build_transformer_decoder(
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=self.embed_dim * 2,
            drop_rate=drop_rate
        )
        self.proj2 = nn.Linear(embed_dim, in_dim)

    def build_transformer_decoder(
            self, num_encoder_layers=3, dim_feedforward=2048, drop_rate=0.1
    ):
        encoder_layer = TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dim_feedforward=dim_feedforward,
            dropout=drop_rate,
            activation='gelu',
            batch_first=True
        )
        encoder_norm = LayerNorm(self.embed_dim)
        encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        return encoder

    def forward(self, x):
        x = self.proj1(x)
        len_x_shape = len(x.shape)
        if len_x_shape == 2:
            x = x.unsqueeze(0)
        B, N, C = x.shape
        x = self.transformer_decoder(x)
        if len_x_shape == 2:
            x = x.squeeze(0)
        x = self.proj2(x)
        return x

