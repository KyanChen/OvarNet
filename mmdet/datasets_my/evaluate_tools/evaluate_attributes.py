import json
import numpy as np

from sklearn.metrics import average_precision_score
import warnings
from sklearn.exceptions import UndefinedMetricWarning
# warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", module="sklearn")

K = 15
def top_K_values(array):
    """Keeps only topK largest values in array.
    """
    indexes = np.argpartition(array, -K, axis=-1)[-K:]  # 第K大的数
    A = set(indexes)
    B = set(list(range(array.shape[0])))
    B -= A
    array[list(B)]=0
    return array


class Evaluator:
    def __init__(
        self,
        fpath_attr2idx,
        fpath_attr_type=None,
        fpath_attr_parent_type=None,
        fpath_attr_headtail=None,
        threshold=0.5,
        exclude_atts=[]
    ):
        # Read file that maps from id to attribute name.
        if isinstance(fpath_attr2idx, str):
            with open(fpath_attr2idx, 'r') as f:
                self.attribute_id_map = json.load(f)
                self.attr2idx = self.attribute_id_map['attribute2id']
                self.idx2attr = self.attribute_id_map['id2attribute']
        else:
            self.attribute_id_map = fpath_attr2idx
            self.attr2idx = self.attribute_id_map['attribute2id']
            self.idx2attr = self.attribute_id_map['id2attribute']

        # # Read file that shows metadata of attributes (e.g., "plaid" is pattern).
        # with open(fpath_attr_type, 'r') as f:
        #     self.attribute_type = json.load(f)
        # with open(fpath_attr_parent_type, 'r') as f:
        #     self.attribute_parent_type = json.load(f)
        #
        # # Read file that shows whether attribute is head/mid/tail.
        # with open(fpath_attr_headtail, 'r') as f:
        #     self.attribute_head_tail = json.load(f)

        self.n_class = len(self.idx2attr)
        self.exclude_atts = exclude_atts
        self.threshold = threshold

        # Cache metric score for each class.
        self.score = {}  # key: i_class -> value: all metrics.
        self.score_topk = {}

    def _clear_cache(self):
        self.score = {}
        self.score_topk = {}

    def get_attr_type(self, attr):
        """Finds type and subtype of a given attribute.
        """
        ty = 'other'
        subty = 'other'
        for x, L in self.attribute_type.items():
            if attr in L:
                subty = x
                break
        for x, L in self.attribute_parent_type.items():
            if subty in L:
                ty = x
                break
        return ty, subty

    def get_attr_head_tail(self, attr):
        """Finds whether attribute is in head/medium/tail group.
        """
        for group, L in self.attribute_head_tail.items():
            if attr in L:
                return group
        assert False, f"Can't find head/medium/tail group for {attr}"

    def evaluate(
        self,
        pred,
        gt_label,
        threshold_type='threshold'
    ):
        """Evaluates a prediction matrix against groundtruth label.

        Args:
        - pred:     prediction matrix [n_instance, n_class].
                    pred[i,j] is the j-th attribute score of instance i-th.
                    These scores should be from 0 -> 1.
        - gt_label: groundtruth label matrix [n_instances, n_class].
                    gt_label[i,j] = 1 if instance i is positively labeled with
                    attribute j, = 0 if it is negatively labeled, and = 2 if
                    it is unlabeled.
        - threshold_type: 'threshold' or 'topk'. 
                          Determines positive vs. negative prediction.
        """
        self.pred = pred
        self.gt_label = gt_label
        self.n_instance = self.gt_label.shape[0]

        # For topK metrics, we keep a version of the prediction matrix that sets
        # non-topK elements as 0 and topK elements as 1.
        P_topk = self.pred.copy()
        P_topk = np.apply_along_axis(top_K_values, 1, P_topk)
        P_topk[P_topk > 0] = 1
        self.pred_topk = P_topk

        all_groups = ['all']
        groups_overall = {
            k: GroupClassMetric(metric_type='overall')
            for k in all_groups
        }
        groups_per_class = {
            k: GroupClassMetric(metric_type='per-class')
            for k in all_groups
        }

        for i_class in range(self.n_class):
            attr = self.idx2attr[repr(i_class)]
            if attr in self.exclude_atts:
                continue

            class_metric = self.get_score_class(i_class, threshold_type=threshold_type)

            # Add to 'all' group.
            groups_overall['all'].add_class(class_metric)
            groups_per_class['all'].add_class(class_metric)

            # # Add to head/medium/tail group.
            # imbalance_group = self.get_attr_head_tail(attr)
            # groups_overall[imbalance_group].add_class(class_metric)
            # groups_per_class[imbalance_group].add_class(class_metric)

            # # Add to corresponding attribute group (color, material, shape, etc.).
            # attr_type, attr_subtype = self.get_attr_type(attr)
            # groups_overall[attr_type].add_class(class_metric)
            # groups_per_class[attr_type].add_class(class_metric)

        # Aggregate final scores.
        # For overall, we're interested in F1.
        # For per-class, we're interested in mean AP, mean recall, mean balanced accuracy.
        scores_overall = {}
        for group_name, group in groups_overall.items():
            scores_overall[group_name] = {
                'f1': group.get_f1(),
                'precision': group.get_precision(),
                'recall': group.get_recall(),
                'tnr': group.get_tnr(),
            }
        scores_per_class = {}
        for group_name, group in groups_per_class.items():
            scores_per_class[group_name] = {
                'ap': group.get_ap(),
                'f1': group.get_f1(),
                'precision': group.get_precision(),
                'recall': group.get_recall(),
                'bacc': group.get_bacc()
            }

        return scores_overall, scores_per_class
        
    def get_score_class(self, i_class, threshold_type='threshold'):
        """Computes all metrics for a given class.

        Args:
        - i_class: class index.
        - threshold_type: 'topk' or 'threshold'. This determines how a
        prediction is positive or negative.
        """
        if threshold_type == 'threshold':
            score = self.score
        else:
            score = self.score_topk
        if i_class in score:
            return score[i_class]

        if threshold_type == 'threshold':
            pred = self.pred[:, i_class].copy()
        else:
            pred = self.pred_topk[:, i_class].copy()
        gt_label = self.gt_label[:, i_class].copy()

        # Find instances that are explicitly labeled (either positive or negative).
        mask_labeled = (gt_label < 2)
        if mask_labeled.sum() == 0:
            # None of the instances have label for this class.
            # assert False, f"0 labeled instances for attribute {self.idx2attr[i_class]}"
            pass
        else:
            # Select ony the labeled ones.
            pred = pred[mask_labeled]
            gt_label = gt_label[mask_labeled]

        if threshold_type == 'threshold':
            # Only computes AP when threshold_type is 'threshold'. This is because when
            # threshold_type is 'topk', pred is a binary matrix.
            ap = average_precision_score(gt_label, pred)

            # Make pred into binary matrix.
            pred[pred > self.threshold] = 1
            pred[pred <= self.threshold] = 0

        class_metric = SingleClassMetric(pred, gt_label)
        if threshold_type == 'threshold':
            class_metric.ap = ap

        # Cache results.
        score[i_class] = class_metric
        
        return class_metric


class GroupClassMetric(object):
    def __init__(self, metric_type):
        """This class computes all metrics for a group of attributes.

        Args:
        - metric_type: 'overall' or 'per-class'.
        """
        self.metric_type = metric_type

        if metric_type == 'overall':
            # Keep track of all stats.
            self.true_pos = 0
            self.false_pos = 0
            self.true_neg = 0
            self.false_neg = 0
            self.n_pos = 0
            self.n_neg = 0
        else:
            self.metric = {
                name: []
                for name in ['recall', 'tnr', 'acc', 'bacc', 'precision', 'f1', 'ap']
            }

    def add_class(self, class_metric):
        """Adds computed metrics of a class into this group.
        """
        if self.metric_type == 'overall':
            self.true_pos += class_metric.true_pos
            self.false_pos += class_metric.false_pos
            self.true_neg += class_metric.true_neg
            self.false_neg += class_metric.false_neg
            self.n_pos += class_metric.n_pos
            self.n_neg += class_metric.n_neg
        else:
            self.metric['recall'].append(class_metric.get_recall())
            self.metric['tnr'].append(class_metric.get_tnr())
            self.metric['acc'].append(class_metric.get_acc())
            self.metric['bacc'].append(class_metric.get_bacc())
            self.metric['precision'].append(class_metric.get_precision())
            self.metric['f1'].append(class_metric.get_f1())
            self.metric['ap'].append(class_metric.ap)

    def get_recall(self):
        """Computes recall.
        """
        if self.metric_type == 'overall':
            n_pos_pred = self.true_pos + self.false_pos
            if n_pos_pred == 0:
                # Model makes 0 positive prediction.
                # This is a special case: we fall back to precision = 1 and recall = 0.
                return 0

            if self.n_pos > 0:
                return self.true_pos / self.n_pos
            return -1
        else:
            if -1 not in self.metric['recall']:
                return np.mean(self.metric['recall'])
            return -1

    def get_tnr(self):
        """Computes true negative rate.
        """
        if self.metric_type == 'overall':
            if self.n_neg > 0:
                return self.true_neg / self.n_neg
            return -1
        else:
            if -1 not in self.metric['tnr']:
                return np.mean(self.metric['tnr'])
            return -1

    def get_acc(self):
        """Computes accuracy.
        """
        if self.metric_type == 'overall':
            if self.n_pos + self.n_neg > 0:
                return (self.true_pos + self.true_neg) / (self.n_pos + self.n_neg)
            return -1
        else:
            if -1 not in self.metric['acc']:
                return np.mean(self.metric['acc'])
            return -1

    def get_bacc(self):
        """Computes balanced accuracy.
        """
        if self.metric_type == 'overall':
            recall = self.get_recall()
            tnr = self.get_tnr()
            if recall == -1 or tnr == -1:
                return -1
            return (recall + tnr) / 2.0
        else:
            if -1 not in self.metric['bacc']:
                return np.mean(self.metric['bacc'])
            return -1

    def get_precision(self):
        """Computes precision.
        """
        if self.metric_type == 'overall':
            n_pos_pred = self.true_pos + self.false_pos
            if n_pos_pred == 0:
                # Model makes 0 positive prediction.
                # This is a special case: we fall back to precision = 1 and recall = 0.
                return 1
            return self.true_pos / n_pos_pred
        else:
            if -1 not in self.metric['precision']:
                return np.mean(self.metric['precision'])
            return -1

    def get_f1(self):
        """Computes F1.
        """
        if self.metric_type == 'overall':
            recall = self.get_recall()
            precision = self.get_precision()
            if precision + recall > 0:
                return 2 * precision * recall / (precision + recall)
            return 0
        else:
            if -1 not in self.metric['f1']:
                return np.mean(self.metric['f1'])
            return -1

    def get_ap(self):
        """Computes mAP.
        """
        assert self.metric_type == 'per-class'
        return np.mean(self.metric['ap'])


class SingleClassMetric(object):
    def __init__(self, pred, gt_label):
        """This class computes all metrics for a single attribute.

        Args:
        - pred: np.array of shape [n_instance] -> binary prediction.
        - gt_label: np.array of shape [n_instance] -> groundtruth binary label.
        """
        if pred is None or gt_label is None:
            self.true_pos = 0
            self.false_pos = 0
            self.true_neg = 0
            self.false_neg = 0
            self.n_pos = 0
            self.n_neg = 0
            self.ap = -1
            return

        self.true_pos = ((gt_label == 1) & (pred == 1)).sum()
        self.false_pos = ((gt_label == 0) & (pred == 1)).sum()
        self.true_neg = ((gt_label == 0) & (pred == 0)).sum()
        self.false_neg = ((gt_label == 1) & (pred == 0)).sum()

        # Number of groundtruth positives & negatives.
        self.n_pos = self.true_pos + self.false_neg
        self.n_neg = self.false_pos + self.true_neg
        
        # AP score.
        self.ap = -1

    def get_recall(self):
        """Computes recall.
        """
        n_pos_pred = self.true_pos + self.false_pos
        if n_pos_pred == 0:
            # Model makes 0 positive prediction.
            # This is a special case: we fall back to precision = 1 and recall = 0.
            return 0

        if self.n_pos > 0:
            return self.true_pos / self.n_pos
        return -1

    def get_tnr(self):
        """Computes true negative rate.
        """
        if self.n_neg > 0:
            return self.true_neg / self.n_neg
        return -1

    def get_acc(self):
        """Computes accuracy.
        """
        if self.n_pos + self.n_neg > 0:
            return (self.true_pos + self.true_neg) / (self.n_pos + self.n_neg)
        return -1

    def get_bacc(self):
        """Computes balanced accuracy.
        """
        recall = self.get_recall()
        tnr = self.get_tnr()
        if recall == -1 or tnr == -1:
            return -1
        return (recall + tnr) / 2.0

    def get_precision(self):
        """Computes precision.
        """
        n_pos_pred = self.true_pos + self.false_pos
        if n_pos_pred == 0:
            # Model makes 0 positive prediction.
            # This is a special case: we fall back to precision = 1 and recall = 0.
            return 1
        return self.true_pos / n_pos_pred

    def get_f1(self):
        """Computes F1.
        """
        recall = self.get_recall()
        precision = self.get_precision()
        
        if precision + recall > 0:
            return 2 * precision * recall / (precision + recall)
        return 0

if __name__ == '__main__':
    attribute_id_map = {'attribute2id': {'0a': 0, '1b': 1, '2a': 2}, 'id2attribute': {'0': '0a', '1': '1a', '2': '2a'}}
    evaluator = Evaluator(attribute_id_map)
    gt_label = np.random.rand(3, 20) > 0.5
    pred_prob = np.random.rand(3, 20)
    scores_overall, scores_per_class = evaluator.evaluate(pred_prob, gt_label)

    scores_overall_topk, scores_per_class_topk = evaluator.evaluate(pred_prob, gt_label, threshold_type='topk')