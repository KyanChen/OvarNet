import torch
from einops import rearrange
from mmcv.cnn import ConvModule
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler, multi_apply, build_bbox_coder
from ..builder import NECKS, build_neck, build_roi_extractor, build_shared_head, HEADS, build_head, build_loss
from mmcv.runner import BaseModule, force_fp32

from ..utils.transformer import PatchMerging


@NECKS.register_module()
class ProposalEncoder(BaseModule):
    def __init__(
            self,
            out_channels=1024,
            bbox_roi_extractor=None,
            shared_head=None,
            attribute_head=None,
            train_cfg=None,
            test_cfg=None,
            pretrained=None,
            init_cfg=None
    ):
        super(ProposalEncoder, self).__init__(init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if shared_head is not None:
            shared_head.pretrained = pretrained
            self.shared_head = build_shared_head(shared_head)

        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.attribute_head = attribute_head
        if attribute_head is None:
            self.bbox_head = self.init_bbox_head(256, out_channels, 14)
            self.feature_proj = nn.Identity()
        else:
            self.bbox_head = build_head(attribute_head)
            self.cls_proj = nn.Sequential(
                nn.Linear(self.bbox_head.embed_dim, self.bbox_head.embed_dim),
                nn.LeakyReLU(),
                nn.Linear(self.bbox_head.embed_dim, out_channels)
            )
            self.reg_proj = nn.Sequential(
                nn.Linear(self.bbox_head.embed_dim, self.bbox_head.embed_dim // 2),
                nn.LeakyReLU(),
                nn.Linear(self.bbox_head.embed_dim // 2, 4)
            )
        self.init_assigner_sampler()

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(self.train_cfg.sampler, context=self)

    def init_bbox_head(self, in_channels, out_channels, roi_size=7):
        if roi_size == 7:
            bbox_head = nn.Sequential(
                ConvModule(
                    in_channels,
                    in_channels,
                    3,
                    stride=2,
                    padding=1,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='ReLU')
                ),
                ConvModule(
                    in_channels,
                    in_channels*2,
                    3,
                    stride=2,
                    padding=1,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='ReLU')
                ),
                ConvModule(
                    in_channels * 2,
                    in_channels * 4,
                    3,
                    stride=2,
                    padding=1,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='ReLU')
                ),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_channels * 4, out_channels, kernel_size=1),
            )
        elif roi_size == 14:
            bbox_head = nn.Sequential(
                ConvModule(
                    in_channels,
                    in_channels*2,
                    3,
                    stride=2,
                    padding=1,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='ReLU')
                ),
                ConvModule(
                    in_channels*2,
                    in_channels*2,
                    3,
                    stride=2,
                    padding=1,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='ReLU')
                ),
                ConvModule(
                    in_channels * 2,
                    in_channels * 2,
                    3,
                    stride=2,
                    padding=1,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='ReLU')
                ),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_channels * 2, out_channels, kernel_size=1),
            )
        else:
            raise NotImplementedError
        # bbox_head = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(in_channels, out_channels, 1)
        # )
        return bbox_head

    def forward(self, x, proposal_list, **kwargs):
        rois = bbox2roi(proposal_list)
        # x 由大到小
        bbox_feats = self.bbox_roi_extractor(x[-self.bbox_roi_extractor.num_inputs:], rois)  # N 256 7 7
        if hasattr(self, 'shared_head'):
            bbox_feats = self.shared_head(bbox_feats)
        proposal_features = self.bbox_head(bbox_feats, **kwargs)
        if self.attribute_head is None:
            proposal_features = proposal_features.view(-1, 1024)
        proposal_features = self.feature_proj(proposal_features)
        return proposal_features, bbox_feats

    def _bbox_forward_train(
            self, x, sampling_results,
            gt_bboxes, gt_labels, img_metas, **kwargs):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)  # [8192, 256, 14, 14]
        # 下面是之前的，需要注意
        # bbox_feats = self.bbox_roi_extractor(x[-self.bbox_roi_extractor.num_inputs:], rois)  # N 256 7 7
        if hasattr(self, 'shared_head'):
            bbox_feats = self.shared_head(bbox_feats)

        cls_rep, reg_rep = self.bbox_head(bbox_feats, **kwargs)
        cls_rep, reg_rep = self.cls_proj(cls_rep), self.reg_proj(reg_rep)
        bbox_results = dict(
            cls_rep=cls_rep, bbox_pred=reg_rep, bbox_feats=bbox_feats)

        lab_bbox_targets = self.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.train_cfg)
        labels, label_weights, bbox_targets, bbox_weights, pos_flag = lab_bbox_targets

        loss_bbox = self.bbox_head.box_loss(
            bbox_results['cls_rep'],
            bbox_results['bbox_pred'],
            rois,
            *lab_bbox_targets,
            reduction_override='none'
        )

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results, cls_rep, labels, label_weights, pos_flag

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      **kwargs
                      ):

        num_imgs = len(img_metas)
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        losses = dict()
        bbox_results, cls_rep, labels, label_weights, pos_flag = self._bbox_forward_train(
            x, sampling_results,
            gt_bboxes, gt_labels, img_metas
        )

        losses.update(bbox_results['loss_bbox'])

        return losses, cls_rep, labels, label_weights, pos_flag, sampling_results

    def simple_test(
            self,
            x,
            proposal_list,
            img_metas,
            proposals=None,
            rescale=False,
            keep_not_scale=False,
            **kwargs):

        rois = bbox2roi(proposal_list)  # Nx5, img_id+4

        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)  # [8192, 256, 14, 14]
        if hasattr(self, 'shared_head'):
            bbox_feats = self.shared_head(bbox_feats)

        cls_rep, reg_rep = self.bbox_head(bbox_feats, **kwargs)
        cls_rep, reg_rep = self.cls_proj(cls_rep), self.reg_proj(reg_rep)
        bbox_results = dict(
            cls_rep=cls_rep, bbox_pred=reg_rep, bbox_feats=bbox_feats)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_rep = bbox_results['cls_rep']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposal_list)
        rois = rois.split(num_proposals_per_img, 0)
        cls_rep = cls_rep.split(num_proposals_per_img, 0)
        bbox_pred = bbox_pred.split(num_proposals_per_img, 0)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        for i in range(len(proposal_list)):
            det_bbox = self.bbox_head.get_bboxes(
                rois[i],
                cls_rep[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=None)
            det_bboxes.append(det_bbox)

        return det_bboxes, cls_rep

    def test_box_given(
            self,
            x,
            proposal_list,
            img_metas=None,
            proposals=None,
            rescale=False,
            keep_not_scale=False,
            **kwargs
    ):
        rois = bbox2roi(proposal_list)
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)  # [8192, 256, 14, 14]
        # 下面是之前的，需要注意
        # bbox_feats = self.bbox_roi_extractor(x[-self.bbox_roi_extractor.num_inputs:], rois)  # N 256 7 7
        if hasattr(self, 'shared_head'):
            bbox_feats = self.shared_head(bbox_feats)

        cls_rep, reg_rep = self.bbox_head(bbox_feats, **kwargs)
        cls_rep, reg_rep = self.cls_proj(cls_rep), self.reg_proj(reg_rep)
        bbox_results = dict(cls_rep=cls_rep, bbox_pred=reg_rep, bbox_feats=bbox_feats)
        return cls_rep


@HEADS.register_module()
class TransformerAttrHead(BaseModule):
    def __init__(self,
                 in_channel=256,
                 embed_dim=512,
                 num_patches=49,
                 use_abs_pos_embed=True,
                 drop_rate=0.,
                 class_token=True,
                 reg_token=False,
                 with_cate_emb=False,
                 num_encoder_layers=3,
                 global_pool=False,
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     clip_border=True,
                     target_means=[0., 0., 0., 0.],
                     target_stds=[0.1, 0.1, 0.2, 0.2]),
                 reg_class_agnostic=True,
                 reg_decoded_bbox=False,
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 ):
        super(TransformerAttrHead, self).__init__(init_cfg)
        self.embed_dim = embed_dim
        self.use_abs_pos_embed = use_abs_pos_embed
        self.num_patches = num_patches
        self.class_token = class_token
        self.reg_token = reg_token
        self.global_pool = global_pool
        self.in_channel = in_channel

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.reg_decoded_bbox = reg_decoded_bbox
        self.reg_class_agnostic = reg_class_agnostic
        self.loss_bbox = build_loss(loss_bbox)

        if self.class_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        if self.reg_token:
            self.reg_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        if with_cate_emb:
            self.cate_emb_proj = nn.Linear(1024, 512)
        if self.use_abs_pos_embed:
            self.absolute_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, self.in_channel) * .02)

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if num_patches == 14*14:
            self.patch_merger = PatchMerging(in_channel, self.embed_dim)
        elif num_patches == 7*7:
            self.patch_merger = nn.Conv2d(in_channel, self.embed_dim, kernel_size=1)
        else:
            raise NotImplementedError
        self.transformer_decoder = self.build_transformer_decoder(num_encoder_layers=num_encoder_layers, dim_feedforward=self.embed_dim * 2)

    def build_transformer_decoder(
            self, num_encoder_layers=3, dim_feedforward=2048
    ):
        encoder_layer = TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        encoder_norm = LayerNorm(self.embed_dim)
        encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        return encoder

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape

        x = rearrange(x, 'b c h w -> b (h w) c')

        if self.use_abs_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        if self.num_patches == 14*14:
            x, down_hw_shape = self.patch_merger(x, (14, 14))
        else:
            x = self.patch_merger(x)

        if 'cate_emb' in kwargs:
            cate_emb = kwargs.get('cate_emb')
            cate_emb = self.cate_emb_proj(cate_emb)
            cate_emb = rearrange(cate_emb, 'B (N C) -> B N C', N=1)
            x = torch.cat((cate_emb, x), dim=1)

        if self.reg_token is not None:
            reg_tokens = self.reg_token.expand(B, -1, -1)
            x = torch.cat((reg_tokens, x), dim=1)

        if self.class_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.transformer_decoder(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        else:
            cls_rep, reg_rep = x[:, 0], x[:, 1]

        return cls_rep, reg_rep

    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels, cfg):
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Args:
            pos_bboxes (Tensor): Contains all the positive boxes,
                has shape (num_pos, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            neg_bboxes (Tensor): Contains all the negative boxes,
                has shape (num_neg, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_bboxes (Tensor): Contains gt_boxes for
                all positive samples, has shape (num_pos, 4),
                the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_labels (Tensor): Contains gt_labels for
                all positive samples, has shape (num_pos, ).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals
            in a single image. Containing the following Tensors:

                - labels(Tensor): Gt_labels for all proposals, has
                  shape (num_proposals,).
                - label_weights(Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
                - bbox_targets(Tensor):Regression target for all
                  proposals, has shape (num_proposals, 4), the
                  last dimension 4 represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights(Tensor):Regression weights for all
                  proposals, has shape (num_proposals, 4).
        """
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        n_class = pos_gt_labels.size(-1)
        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]

        labels = pos_bboxes.new_full((num_samples, n_class),
                                     n_class,
                                     dtype=torch.long)
        pos_flag = pos_bboxes.new_zeros(num_samples)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[:num_pos, :] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
            pos_flag[:num_pos] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights, pos_flag

    @force_fp32(apply_to=('bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None,
                   keep_not_scale=False):
        """Transform network output for a batch into bbox predictions.

        Args:
            rois (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            img_shape (Sequence[int], optional): Maximum bounds for boxes,
                specifies (H, W, C) or (H, W).
            scale_factor (ndarray): Scale factor of the
               image arrange as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None

        Returns:
            tuple[Tensor, Tensor]:
                First tensor is `det_bboxes`, has the shape
                (num_boxes, 5) and last
                dimension 5 represent (tl_x, tl_y, br_x, br_y, score).
                Second tensor is the labels with shape (num_boxes, ).
        """

        bboxes = self.bbox_coder.decode(
            rois[..., 1:], bbox_pred, max_shape=img_shape)

        if keep_not_scale:
            not_scaled_bboxes = bboxes.clone()
        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
                bboxes.size()[0], -1)

        return bboxes


    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_target_single` function.

        Args:
            sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            gt_bboxes (list[Tensor]): Gt_bboxes of all images in a batch,
                each tensor has shape (num_gt, 4),  the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            gt_labels (list[Tensor]): Gt_labels of all images in a batch,
                each tensor has shape (num_gt,).
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

                - labels (list[Tensor],Tensor): Gt_labels for all
                  proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - label_weights (list[Tensor]): Labels_weights for
                  all proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - bbox_targets (list[Tensor],Tensor): Regression target
                  for all proposals in a batch, each tensor in list
                  has shape (num_proposals, 4) when `concat=False`,
                  otherwise just a single tensor has shape
                  (num_all_proposals, 4), the last dimension 4 represents
                  [tl_x, tl_y, br_x, br_y].
                - bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 4) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 4).
        """

        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [gt_label[res.pos_assigned_gt_inds] for res, gt_label in zip(sampling_results, gt_labels)]
        labels, label_weights, bbox_targets, bbox_weights, pos_flag = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            pos_flag = torch.cat(pos_flag, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights, pos_flag

    @force_fp32(apply_to='bbox_pred')
    def box_loss(
            self,
            cls_score,
            bbox_pred,
            rois,
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            pos_flag,
            reduction_override=None):

        losses = dict()
        if bbox_pred is not None:
            # bg_class_ind = labels.size(-1)
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            # pos_inds = (labels >= 0) & (labels < bg_class_ind)
            pos_inds = pos_flag
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses

