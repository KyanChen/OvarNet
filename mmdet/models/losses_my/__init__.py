from mmdet.models.losses_my.label_smoothed_cross_entropy import AdjustLabelSmoothedCrossEntropyCriterion
from .milloss import MILCrossEntropy

__all__ = [
    'AdjustLabelSmoothedCrossEntropyCriterion',
    'MILCrossEntropy'
]
