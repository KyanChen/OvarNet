import json
from collections import OrderedDict

from ..builder import DETECTORS
from ..detectors.two_stage import TwoStageDetector
import warnings
from mmcv.runner import get_dist_info
import torch
from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from ..detectors.base import BaseDetector
from ..detectors.faster_rcnn import FasterRCNN


@DETECTORS.register_module()
class FasterRCNNInfer(FasterRCNN):
    # Inference ONLY RPN Network
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(FasterRCNNInfer, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None


    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

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

    def simple_test(self, img, img_metas, proposals=None, rescale=False, img_key=None):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        results = self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

        # import pdb
        # pdb.set_trace()
        proposal_list = [x[0].cpu() for x in results]

        if img_key is not None:
            folder = '/data/kyanchen/prompt/data/CC3M/proposals'
            '''list[Tensor]: Proposals of each image, each item has shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).'''
            for key, proposal in zip(img_key[0], proposal_list):
                proposal = proposal.numpy()
                proposal[:, 2] = proposal[:, 2] - proposal[:, 0]
                proposal[:, 3] = proposal[:, 3] - proposal[:, 1]
                proposal = proposal.tolist()
                saved_data = {'img_key': key, 'proposals': proposal}
                json.dump(saved_data, open(folder+f'/{key}.json', 'w'), indent=4)
        return proposal_list

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def onnx_export(self, img, img_metas):

        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        x = self.extract_feat(img)
        proposals = self.rpn_head.onnx_export(x, img_metas)
        if hasattr(self.roi_head, 'onnx_export'):
            return self.roi_head.onnx_export(x, proposals, img_metas)
        else:
            raise NotImplementedError(
                f'{self.__class__.__name__} can not '
                f'be exported to ONNX. Please refer to the '
                f'list of supported models,'
                f'https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html#list-of-supported-models-exportable-to-onnx'
                # noqa E501
            )

