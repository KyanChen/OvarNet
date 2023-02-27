from abc import abstractmethod
from collections import OrderedDict
import torch.distributed as dist

from mmcv.parallel import DataContainer
from mmcv.runner import auto_fp16

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from ..detectors.base import BaseDetector
import warnings
import torch


@DETECTORS.register_module()
class MaskRCNNCLIP(BaseDetector):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 proposal_encoder,
                 attribute_encoder,
                 attribute_pred_head,
                 train_cfg,
                 test_cfg,
                 with_proposal_ann=False,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(MaskRCNNCLIP, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.with_proposal_ann = with_proposal_ann

        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.proposal_encoder = build_neck(proposal_encoder)
        self.attribute_encoder = build_neck(attribute_encoder)
        self.attribute_pred_head = build_head(attribute_pred_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for name, module in self.named_children():
            if name in ['backbone', 'neck', 'rpn_head', 'roi_head']:
                module.train(False)
            else:
                module.train(mode)
        return self

    def aug_test(self, imgs, img_metas, **kwargs):
        pass

    def forward_test(self, imgs, img_metas, gt_bboxes, **kwargs):
        # import pdb
        # pdb.set_trace()
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                # raise TypeError(f'{name} must be a list, but got {type(var)}')
                imgs = [imgs]
                img_metas = [img_metas]
                gt_bboxes = [gt_bboxes]


        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
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
            return self.simple_test(imgs[0], img_metas[0], gt_bboxes[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img',))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0])

        if return_loss:
            if self.with_proposal_ann:
                return self.forward_train_with_proposal_ann(img, img_metas, **kwargs)
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
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

    def train_step(self, data, optimizer):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def val_step(self, data, optimizer=None):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs,)
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs,)
        return outs

    def select_proposal_list(self, det_labels, not_scaled_boxes, category_attribute_pair, img_metas):
        returned_proposal_list = []
        returned_proposal_att = []
        for img_id, img_category_attribute_pair in enumerate(category_attribute_pair):
            det_label = det_labels[img_id]
            select_proposal = []
            select_att = []
            for idx, categories_id in enumerate(img_category_attribute_pair.keys()):
                select_proposals = not_scaled_boxes[img_id][det_label == categories_id]
                if len(select_proposals):
                    select_proposal.append(select_proposals[0:1])  # 取置信度最大的一个
                    select_att.append(img_category_attribute_pair[categories_id])
            if len(select_proposal):
                select_proposal = torch.cat(select_proposal)
            else:
                select_proposal = not_scaled_boxes[0].new_zeros(0, 5)
            returned_proposal_list.append(select_proposal)
            returned_proposal_att.append(select_att)

        return returned_proposal_list, returned_proposal_att

    def forward_train(self, img, img_metas, category_attribute_pair, **kwargs):
        if isinstance(img, DataContainer):
            img = img.data[0]
            img_metas = img_metas.data[0]
            category_attribute_pair = category_attribute_pair.data[0]
        with torch.no_grad():
            x = self.extract_feat(img)  # 5x[Bx256xHxW]
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)  # Bx[tensor(Nx5)]

            results, det_labels, not_scaled_boxes = self.roi_head.simple_test(
                x, proposal_list, img_metas, rescale=True, keep_not_scale=True
            )

        # select box proposals
        att_proposal_list, proposal_att_list = self.select_proposal_list(
            det_labels, not_scaled_boxes, category_attribute_pair, img_metas
        )  # Bx[tensor(Nx5)]

        proposal_flatten_features, bbox_feats = self.proposal_encoder(x, att_proposal_list)

        attribute_idxs = torch.cat([single_proposal_att for single_img_atts in proposal_att_list for single_proposal_att in single_img_atts])
        unique_attribute_idxs = torch.unique(attribute_idxs)
        neg_attribute_idxs = torch.randperm(self.attribute_encoder.num_attributes)[:len(unique_attribute_idxs)]
        unique_attribute_idxs = torch.unique(torch.cat((unique_attribute_idxs, neg_attribute_idxs)))
        proposal_attribute_features = self.attribute_encoder.forward_train(unique_attribute_idxs, device=img.device)  # 6x1024
        proposal_flatten_features = proposal_flatten_features.squeeze(dim=-1).squeeze(dim=-1)  # 2x1024

        losses = {}
        loss, logits_per_image, logits_per_text = self.attribute_pred_head.forward_train(
            proposal_flatten_features, proposal_attribute_features, proposal_att_list, unique_attribute_idxs
        )

        losses.update(loss)
        return losses

    def forward_train_with_proposal_ann(self, img, img_metas, gt_bboxes, attrs, **kwargs):
        with torch.no_grad():
            x = self.extract_feat(img)  # 5x[Bx256xHxW]

        proposal_flatten_features, bbox_feats = self.proposal_encoder(x, gt_bboxes)
        proposal_flatten_features = proposal_flatten_features.squeeze(dim=-1).squeeze(dim=-1)  # 2x1024

        attribute_idxs = torch.nonzero(attrs)
        proposal_att_list = []
        for proposal_idx in range(len(gt_bboxes)):
            proposal_att_list.append([attribute_idxs[attribute_idxs[:, 0] == proposal_idx][:, 1]])

        unique_attribute_idxs = torch.unique(attribute_idxs[:, 1])
        neg_attribute_idxs = torch.randperm(self.attribute_encoder.num_attributes)[:len(unique_attribute_idxs)*1].to(img.device)
        unique_attribute_idxs = torch.unique(torch.cat((unique_attribute_idxs, neg_attribute_idxs)))
        # unique_attribute_idxs = torch.arange(self.attribute_encoder.num_attributes).to(img.device)
        proposal_attribute_features = self.attribute_encoder.forward_train(unique_attribute_idxs, device=img.device)  # 6x1024


        losses = {}
        loss, logits_per_image, logits_per_text = self.attribute_pred_head.forward_train(
            proposal_flatten_features, proposal_attribute_features, proposal_att_list, unique_attribute_idxs
        )

        losses.update(loss)
        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, gt_bboxes, **kwargs):
        """Test without augmentation."""
        # img: 2 3 800 1216, gt_bboxes 2*list[tensor(1,4)]
        x = self.extract_feat(img)  # 5x[Bx256xHxW]
        # proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)  # Bx[tensor(Nx5)]
        #
        # results, det_labels, not_scaled_boxes = self.roi_head.simple_test(
        #     x, proposal_list, img_metas, rescale=True, keep_not_scale=True
        # )
        #
        # # select box proposals
        # att_proposal_list, proposal_att_list = self.select_proposal_list(
        #     det_labels, not_scaled_boxes, category_attribute_pair, img_metas
        # )  # Bx[tensor(Nx5)]

        proposal_flatten_features, bbox_feats = self.proposal_encoder(x, gt_bboxes)
        proposal_flatten_features = proposal_flatten_features.squeeze(dim=-1).squeeze(dim=-1)  # 2x1024

        unique_attribute_idxs = torch.arange(self.attribute_encoder.num_attributes)
        proposal_attribute_features = self.attribute_encoder.forward_train(unique_attribute_idxs, device=img.device)  # 204x1024


        logits_per_image = self.attribute_pred_head.simple_test(
            proposal_flatten_features, proposal_attribute_features
        )

        return logits_per_image

