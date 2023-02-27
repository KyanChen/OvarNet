import copy
import inspect
import math
import warnings
import random

import cv2
import mmcv
import numpy as np
from ...datasets.builder import PIPELINES


@PIPELINES.register_module()
class ScaleCrop:
    """
    crop img according to boxes
    """

    def __init__(self, scale_range=[0.1, 0.4]):
        self.scale_range = scale_range

    def _get_crop_size(self, instance_size):
        h, w = instance_size
        random_scale = self.scale_range[0] + np.random.rand() * (self.scale_range[1] - self.scale_range[0])
        random_scale += 1
        return int(h * random_scale + 0.5), int(w * random_scale + 0.5)

    def __call__(self, results):
        img = results['img']
        l, t, r, b = results['crop_box']
        instance_size = [b-t, r-l]  # H,W
        crop_size = self._get_crop_size(instance_size)
        cx = (l+r)/2.
        cy = (t+b)/2.
        y0 = cy - crop_size[0]/2
        y1 = cy + crop_size[0]/2
        x0 = cx - crop_size[1]/2
        x1 = cx + crop_size[1]/2
        y0, y1 = np.clip([y0, y1], 0, img.shape[0]).astype(np.int)
        x0, x1 = np.clip([x0, x1], 0, img.shape[1]).astype(np.int)
        img = img[y0:y1, x0:x1, ...]
        results['img'] = img
        results['img_shape'] = img.shape
        return results


@PIPELINES.register_module()
class RandomExpandAndCropBox:
    """
    random expand box and crop box
    """

    def __init__(self, expand_range=(1, 1.2), crop_range=(0.6, 1), center_crop=False, prob=0.5):
        self.prob = prob
        self.expand_min_ratio, self.expand_max_ratio = expand_range
        self.center_crop = center_crop
        self.crop_min_ratio, self.crop_max_ratio = crop_range

    def __call__(self, results):
        if random.uniform(0, 1) > self.prob:
            return results
        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        img = results['img']

        h, w = img.shape[:2]
        # expand bboxes
        for key in results.get('bbox_fields', []):
            bboxes = results[key]
            # import pdb
            # pdb.set_trace()
            for i in range(len(bboxes)):
                ratio = random.uniform(self.expand_min_ratio, self.expand_max_ratio)
                cxcy = (bboxes[i, 0:2] + bboxes[i, 2:4]) / 2
                expand_wh = ratio * (bboxes[i, 2:4] - bboxes[i, 0:2])
                lt = cxcy - expand_wh / 2
                rb = cxcy + expand_wh / 2
                bboxes[i, ...] = np.concatenate((lt, rb), axis=0)
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, w)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, h)
            results[key] = bboxes

        # crop bboxes
        for key in results.get('bbox_fields', []):
            bboxes = results[key]
            for i in range(len(bboxes)):
                box_wh = bboxes[i, 2:4] - bboxes[i, 0:2]
                crop_ratio = np.array([
                    random.uniform(self.crop_min_ratio, self.crop_max_ratio),
                    random.uniform(self.crop_min_ratio, self.crop_max_ratio)])
                crop_size = box_wh * crop_ratio

                margin_w = max(box_wh[0] - crop_size[0], 0)
                margin_h = max(box_wh[1] - crop_size[1], 0)
                offset_w = np.random.randint(0, margin_w + 1)
                offset_h = np.random.randint(0, margin_h + 1)
                crop_x1, crop_x2 = offset_w, offset_w + crop_size[0]
                crop_y1, crop_y2 = offset_h, offset_h + crop_size[1]
                bboxes[i, 0:2] += np.array([offset_w, offset_h])
                bboxes[i, 2:] = bboxes[i, 0:2] + np.asarray(crop_size)

            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, w)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, h)
            results[key] = bboxes
        return results
