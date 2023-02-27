# Copyright (c) OpenMMLab. All rights reserved.
# from .coco_api_ori import COCO, COCOeval
from .panoptic_evaluation_ori import pq_compute_multi_core, pq_compute_single_core

from .coco import COCO
from .cocoeval import COCOeval
from .mask import _mask as maskUtils
__all__ = [
    'COCO', 'COCOeval', 'maskUtils'
    'pq_compute_multi_core', 'pq_compute_single_core'
]
