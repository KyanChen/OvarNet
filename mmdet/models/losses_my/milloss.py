import torch
import torch.nn as nn
from ..builder import LOSSES
import torch.nn.functional as F

@LOSSES.register_module()
class MILCrossEntropy(nn.Module):
    """
    Multi-instance learning loss
    """
    def __init__(self):
        super(MILCrossEntropy, self).__init__()

    def forward(self, pred_logits, target, dim=-1, weighted_unk=False, weights=None, avg_positives=False):
        # # for numerical stability
        # logits_max, _ = torch.max(x, dim=1, keepdim=True)
        # logits = x - logits_max.detach()
        # exp_logits = torch.exp(logits)
        #
        # # get non-zero entries off-diagonal
        # # identity = torch.eye(target.shape[0]).type_as(target)
        # # laplacian = 1 - (target - identity)
        # probs = exp_logits / (exp_logits).sum(dim=dim, keepdim=True)
        if weighted_unk:
            pred_logits[target == 2] /= weighted_unk
            target[target == 2] = 0
        probs = F.softmax(pred_logits, dim=-1)
        # 对于没有标签的样本忽视

        valid_mask = torch.any(target > 0, dim=-1)
        probs = probs[valid_mask]
        target = target[valid_mask]
        if len(target) == 0:
            return torch.tensor(0., device=pred_logits.device)
        if avg_positives:  # average the logits over positive targets
            loss = -torch.log(torch.sum(target * probs, dim=dim) / (torch.sum(target, dim=dim) + 1e-6))
        else:  # sum the logits over positive targets
            loss = -torch.log(torch.sum(target * probs, dim=dim))
        if weights is not None:
            return (loss * weights).mean()
        return loss.mean()