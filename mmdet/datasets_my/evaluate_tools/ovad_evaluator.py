"""
API for evaluating OVAD Dataset: 
OVAD API is a Python API that assists in evaluating the annotations in OVAD. 
This API was made based on VAW evaluation API:
https://github.com/adobe-research/vaw_dataset/blob/main/eval/evaluator.py
"""

import os
import copy
import math
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from tabulate import tabulate
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    PrecisionRecallDisplay,
)

def top_K_values(array, K=15):
    """Keeps only topK largest values in array."""
    indexes = np.argpartition(array, -K, axis=-1)[-K:]
    A = set(indexes)
    B = set(list(range(array.shape[0])))
    B -= A
    array[list(B)] = 0
    return array


class AttEvaluator(object):
    def __init__(
        self,
        attr2idx,
        attr_type=None,
        attr_parent_type=None,
        attr_headtail=None,
        att_seen_unseen=None,
        dataset_name="",
        threshold=0.5,
        top_k=15,
        exclude_atts=[],
        output_file="",
    ):
        """Initializes evaluator for attribute prediction.

        Args:
        - attr2idx: attribute class index file.
        - attr_type: attribute type file.
        - attr_parent_type: attribute parent type file.
        - attr_headtail: attribute head/mid/tail categorization file.
        - att_seen_unseen: attribute seen/unseen categorization file.
        - threshold: positive/negative threshold (for Accuracy metric).
        - exclude_atts: any attribute classes to be excluded from evaluation.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Building Attribute Evaluator of {}.".format(dataset_name))
        self.dataset_name = dataset_name

        # Read file that maps from id to attribute name.
        self.attr2idx = attr2idx
        self.n_class = len(self.attr2idx)
        self.idx2attr = {v: k for k, v in self.attr2idx.items()}

        # Read file that shows metadata of attributes (e.g., "plaid" is pattern).
        self.attribute_type = attr_type if attr_type is not None else {}
        self.attribute_parent_type = (
            attr_parent_type if attr_parent_type is not None else {}
        )

        # Read file that shows whether attribute is head/mid/tail.
        self.attribute_head_tail = attr_headtail if attr_headtail is not None else {}

        # Read file that shows whether attribute is seen/unseen.
        self.attribute_seen_unseen = (
            att_seen_unseen if att_seen_unseen is not None else {}
        )

        self.all_groups = (
            ["all"]
            + list(self.attribute_head_tail.keys())
            + list(self.attribute_parent_type.keys())
            + list(self.attribute_seen_unseen.keys())
        )

        self.exclude_atts = exclude_atts
        self.threshold = threshold
        self.top_k = top_k
        self.output_file = output_file

        # Cache metric score for each class.
        self.score = {}  # key: i_class -> value: all metrics.
        self.score_topk = {}
        self.scores_overall = {}
        self.scores_per_class = {}

        # Keep predition and ground truth vectors
        self.prediction = None
        self.ground_truth = None
        self.instance_info = None

    def register_gt_pred_vectors(self, gt, pred, instance_info):
        assert (
            pred.shape == gt.shape
        ), "GT ({}) and prediction ({}) have to be the same shape".format(
            gt.shape, pred.shape
        )
        self.ground_truth = gt
        self.prediction = pred
        self.instance_info = instance_info

    def _clear_cache(self):
        self.score = {}
        self.score_topk = {}
        self.scores_overall = {}
        self.scores_per_class = {}

    def get_attr_type(self, attr):
        """Finds type and subtype of a given attribute."""
        ty = "other"
        subty = "other"
        for x, L in self.attribute_type.items():
            if attr in L:
                subty = x
                break
        for x, L in self.attribute_parent_type.items():
            if subty in L:
                ty = x
                break
        return ty, subty

    def get_attr_seen_unseen(self, attr):
        """Finds whether attribute is in head/medium/tail group."""
        for group, L in self.attribute_seen_unseen.items():
            if attr in L:
                return group
        assert False, f"Can't find seen/unseen group for {attr}"

    def get_attr_head_tail(self, attr):
        """Finds whether attribute is in head/medium/tail group."""
        for group, L in self.attribute_head_tail.items():
            if attr in L:
                return group
        assert False, f"Can't find head/medium/tail group for {attr}"

    def evaluate(self, pred, gt_label, threshold_type="threshold"):
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
        if self.pred.max() > 1.0 or self.pred.min() < 0.0:
            self.logger.warning(
                "Predictions values are out of range [0, 1], "
                + "range obtained [{:.2}, {:.2}]. Clipping to range.".format(
                    self.pred.min(), self.pred.max()
                )
            )
            self.pred[self.pred < 0.0] = 0.0
            self.pred[self.pred > 1.0] = 1.0
        self.gt_label = gt_label
        self.n_instance = self.gt_label.shape[0]

        # For topK metrics, we keep a version of the prediction matrix that sets
        # non-topK elements as 0 and topK elements as 1.
        P_topk = self.pred.copy()
        P_topk = np.apply_along_axis(top_K_values, 1, P_topk, self.top_k)
        P_topk[P_topk > 0] = 1
        self.pred_topk = P_topk

        groups_overall = {
            k: GroupClassMetric(metric_type="overall") for k in self.all_groups
        }
        groups_per_class = {
            k: GroupClassMetric(metric_type="per-class") for k in self.all_groups
        }

        for i_class in range(self.n_class):
            attr = self.idx2attr[i_class]
            if attr in self.exclude_atts:
                continue

            class_metric = self.get_score_class(i_class, threshold_type=threshold_type)

            # Add to 'all' group.
            groups_overall["all"].add_class(class_metric)
            groups_per_class["all"].add_class(class_metric)

            # Add to head/medium/tail group.
            if len(self.attribute_head_tail) > 0:
                imbalance_group = self.get_attr_head_tail(attr)
                groups_overall[imbalance_group].add_class(class_metric)
                groups_per_class[imbalance_group].add_class(class_metric)

            # Add to corresponding attribute group (color, material, shape, etc.).
            if len(self.attribute_parent_type) > 0:
                attr_type, attr_subtype = self.get_attr_type(attr)
                groups_overall[attr_type].add_class(class_metric)
                groups_per_class[attr_type].add_class(class_metric)

            # Add to seen/unseen group.
            if len(self.attribute_seen_unseen) > 0:
                seen_unseen_group = self.get_attr_seen_unseen(attr)
                groups_overall[seen_unseen_group].add_class(class_metric)
                groups_per_class[seen_unseen_group].add_class(class_metric)

        # Aggregate final scores.
        # For overall, we're interested in F1.
        # For per-class, we're interested in mean AP, mean recall, mean balanced accuracy.
        scores_overall = {}
        for group_name, group in groups_overall.items():
            scores_overall[group_name] = {
                "f1": group.get_f1(),
                "precision": group.get_precision(),
                "recall": group.get_recall(),
                "tnr": group.get_tnr(),
                "rand_precision": group.get_random_precision(),
            }
        self.scores_overall[threshold_type] = scores_overall
        scores_per_class = {}
        for group_name, group in groups_per_class.items():
            scores_per_class[group_name] = {
                "ap": group.get_ap(),
                "f1": group.get_f1(),
                "precision": group.get_precision(),
                "recall": group.get_recall(),
                "bacc": group.get_bacc(),
                "rand_precision": group.get_random_precision(),
            }
        self.scores_per_class[threshold_type] = scores_per_class

        return scores_overall, scores_per_class

    def get_score_class(self, i_class, threshold_type="threshold"):
        """Computes all metrics for a given class.

        Args:
        - i_class: class index.
        - threshold_type: 'topk' or 'threshold'. This determines how a
        prediction is positive or negative.
        """
        if threshold_type == "threshold":
            score = self.score
        else:
            score = self.score_topk
        if i_class in score:
            return score[i_class]

        if threshold_type == "threshold":
            pred = self.pred[:, i_class].copy()
        else:
            pred = self.pred_topk[:, i_class].copy()
        gt_label = self.gt_label[:, i_class].copy()

        # Find instances that are explicitly labeled (either positive or negative).
        mask_labeled = gt_label < 2
        if mask_labeled.sum() == 0:
            # None of the instances have label for this class.
            # assert False, f"0 labeled instances for attribute {self.idx2attr[i_class]}"
            pass
        else:
            # Select ony the labeled ones.
            pred = pred[mask_labeled]
            gt_label = gt_label[mask_labeled]

        if threshold_type == "threshold":
            # Only computes AP when threshold_type is 'threshold'. This is because when
            # threshold_type is 'topk', pred is a binary matrix.
            # Only compute for classes which have at least one positive instance
            if (gt_label == 1).sum() > 0:
                ap = average_precision_score(gt_label, pred)
                # print('i_class', i_class)
                # print('gt_label', gt_label)
                # print('pred', pred)
                # print('ap', ap)
                precision, recall, thresholds = precision_recall_curve(gt_label, pred)
                pr_curve = {"p": precision, "r": recall, "t": thresholds}
                assert not math.isnan(ap)
            else:
                ap = float("nan")
                pr_curve = {"p": [], "r": [], "t": []}

            # Make pred into binary matrix.
            pred[pred > self.threshold] = 1
            pred[pred <= self.threshold] = 0

        class_metric = SingleClassMetric(pred, gt_label)
        if threshold_type == "threshold":
            class_metric.ap = ap
            class_metric.pr_curve = pr_curve

        # Cache results.
        score[i_class] = class_metric

        return class_metric

    def apply_percentage(self, obj):
        if isinstance(obj, dict):  # if dict, apply to each key
            return {k: self.apply_percentage(v) for k, v in obj.items()}
        elif isinstance(obj, list):  # if list, apply to each element
            return [self.apply_percentage(elem) for elem in obj]
        else:
            if isinstance(obj, (int, float)):
                return obj * 100
            return obj

    def plot_pr_curves(self, save_dir, category):
        if category in self.attribute_parent_type.keys():
            all_att_type = []
            for subtype in self.attribute_parent_type[category]:
                all_att_type += list(self.attribute_type[subtype])
            colors = cm.rainbow(np.linspace(0, 1, len(all_att_type)))
            _, ax = plt.subplots(figsize=(7, 8))
            max_r = 0
            max_p = 0

            f_scores = np.linspace(0.2, 0.8, num=4)
            lines, labels = [], []
            for f_score in f_scores:
                x = np.linspace(0.01, 1)
                y = f_score * x / (2 * x - f_score)
                (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
                plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

            for att, color in zip(all_att_type, colors):
                attr_type, attr_subtype = self.get_attr_type(att)
                att_idx = self.attr2idx[att]
                pr_curve = self.score[att_idx].pr_curve
                ap = self.score[att_idx].ap
                rand_precision = self.score[att_idx].rand_precision
                if (
                    attr_type != category
                    or len(pr_curve["r"]) == 0
                    or len(pr_curve["p"]) == 0
                    or ap == -1
                ):
                    continue
                max_r = max(max_r, max(pr_curve["r"]))
                max_p = max(max_p, max(pr_curve["p"]))
                display = PrecisionRecallDisplay(
                    recall=pr_curve["r"],
                    precision=pr_curve["p"],
                    average_precision=ap,
                )
                display.plot(ax=ax, name=f"{att}", color=color)
                plt.plot(0.5, rand_precision, color=color, marker="o")

            if max_r == 0 or max_p == 0:
                return

            # add the legend for the iso-f1 curves
            handles, labels = display.ax_.get_legend_handles_labels()
            handles.extend([l])
            labels.extend(["iso-f1 curves"])
            # set the legend and the axes
            ax.set_xlim([0.0, max_r])
            ax.set_ylim([0.0, max_p])
            ax.legend(handles=handles, labels=labels, loc="best")
            ax.set_title("Precision-Recall curve {}".format(category))
            if max_r > 0 or max_p > 0:
                path_plot = os.path.join(save_dir, "pr_curves_{}.png".format(category))
                print("Saving pr plot {}".format(path_plot))
                plt.savefig(path_plot)
        plt.close("all")

    def plot_pr_curves_types(self, save_dir, category):
        if category in self.attribute_parent_type.keys():
            for subtype in self.attribute_parent_type[category]:
                all_att_type = self.attribute_type[subtype]
                colors = cm.rainbow(np.linspace(0, 1, len(all_att_type)))
                _, ax = plt.subplots(figsize=(7, 8))
                max_r = 0
                max_p = 0

                f_scores = np.linspace(0.2, 0.8, num=4)
                lines, labels = [], []
                for f_score in f_scores:
                    x = np.linspace(0.01, 1)
                    y = f_score * x / (2 * x - f_score)
                    (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
                    plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

                for att, color in zip(all_att_type, colors):
                    attr_type, attr_subtype = self.get_attr_type(att)
                    att_idx = self.attr2idx[att]
                    pr_curve = self.score[att_idx].pr_curve
                    ap = self.score[att_idx].ap
                    rand_precision = self.score[att_idx].rand_precision
                    if (
                        attr_type != category
                        or len(pr_curve["r"]) == 0
                        or len(pr_curve["p"]) == 0
                        or ap == -1
                    ):
                        continue
                    max_r = max(max_r, max(pr_curve["r"]))
                    max_p = max(max_p, max(pr_curve["p"]))
                    display = PrecisionRecallDisplay(
                        recall=pr_curve["r"],
                        precision=pr_curve["p"],
                        average_precision=ap,
                    )
                    display.plot(ax=ax, name=f"{att}", color=color)
                    plt.plot(0.5, rand_precision, color=color, marker="o")

                if max_r == 0 or max_p == 0:
                    return

                # add the legend for the iso-f1 curves
                handles, labels = display.ax_.get_legend_handles_labels()
                handles.extend([l])
                labels.extend(["iso-f1 curves"])
                # set the legend and the axes
                ax.set_xlim([0.0, max_r])
                ax.set_ylim([0.0, max_p])
                ax.legend(handles=handles, labels=labels, loc="best")
                ax.set_title("Precision-Recall curve {}".format(subtype))
                if max_r > 0 or max_p > 0:
                    path_plot = os.path.join(
                        save_dir, "pr_curves_{}.png".format(subtype)
                    )
                    # print("Saving pr plot {}".format(path_plot))
                    plt.savefig(path_plot)
                plt.close("all")

    def print_evaluation(
        self, pred=None, gt_label=None, output_file="", percentage=True
    ):
        # Compute scores.
        if pred is not None and gt_label is not None:
            orig_pred = pred.copy()
            orig_gt = gt_label.copy().astype("float64")

            # calculate real scores
            self._clear_cache()
            scores_overall, scores_per_class = self.evaluate(
                pred.copy(), gt_label.copy()
            )
            scores_overall_topk, scores_per_class_topk = self.evaluate(
                pred.copy(), gt_label.copy(), threshold_type="topk"
            )
        elif (
            "threshold" in self.scores_overall.keys()
            and "threshold" in self.scores_per_class.keys()
            and "topk" in self.scores_overall.keys()
            and "topk" in self.scores_per_class.keys()
        ):
            scores_overall = self.scores_overall["threshold"]
            scores_per_class = self.scores_per_class["threshold"]
            scores_overall_topk = self.scores_overall["topk"]
            scores_per_class_topk = self.scores_per_class["topk"]
        elif self.prediction is not None and self.ground_truth is not None:
            pred = self.prediction.copy().astype("float64")
            gt_label = self.ground_truth.copy().astype("float64")
            instance_info = copy.deepcopy(self.instance_info)

            # calculate real scores
            self._clear_cache()
            scores_overall, scores_per_class = self.evaluate(
                pred.copy(), gt_label.copy()
            )
            scores_overall_topk, scores_per_class_topk = self.evaluate(
                pred.copy(), gt_label.copy(), threshold_type="topk"
            )
        else:
            assert False, "No predictions to evaluate"

        if percentage:
            scores_overall = self.apply_percentage(scores_overall)
            scores_per_class = self.apply_percentage(scores_per_class)
            scores_overall_topk = self.apply_percentage(scores_overall_topk)
            scores_per_class_topk = self.apply_percentage(scores_per_class_topk)

        results = {
            f"OV_t{self.threshold:.1f}": scores_overall,
            f"PC_t{self.threshold:.1f}": scores_per_class,
            f"OV_@{self.top_k}": scores_overall_topk,
            f"PC_@{self.top_k}": scores_per_class_topk,
        }

        category = 'all'
        print(f"----------{category.upper()}----------")
        print(f"mAP: {scores_per_class[category]['ap']:.4f}")
        print("Per-class (threshold {:.2f}):".format(self.threshold))
        for metric in ["recall", "precision", "f1", "bacc"]:
            if metric in scores_per_class[category]:
                print(
                    f"- {metric}: {scores_per_class[category][metric]:.4f}"
                )

        print("Per-class (top {}):".format(self.top_k))
        for metric in ["recall", "precision", "f1"]:
            if metric in scores_per_class_topk[category]:
                print(
                    f"- {metric}: {scores_per_class_topk[category][metric]:.4f}"
                )

        print("Overall (threshold {:.2f}):".format(self.threshold))
        for metric in ["recall", "precision", "f1", "bacc"]:
            if metric in scores_overall[category]:
                print(
                    f"- {metric}: {scores_overall[category][metric]:.4f}"
                )

        print("Overall (top {}):".format(self.top_k))
        for metric in ["recall", "precision", "f1"]:
            if metric in scores_overall_topk[category]:
                print(
                    f"- {metric}: {scores_overall_topk[category][metric]:.4f}"
                )

        if output_file == "" and self.output_file != "":
            output_file = self.output_file
        if output_file != "":
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            print("Saving results in file: " + output_file)
            print("Dataset: " + output_file)

            CATEGORIES = self.all_groups
            mAP_per_att_flatten = []
            with open(output_file, "a") as f:
                for category in CATEGORIES:
                    mAP_per_att_flatten += [category, scores_per_class[category]["ap"]]
                    f.write(f"----------{category.upper()}----------\n")
                    f.write(f"mAP: {scores_per_class[category]['ap']:.4f}\n")

                    f.write("Per-class (threshold {:.2f}):\n".format(self.threshold))
                    for metric in ["recall", "precision", "f1", "bacc"]:
                        if metric in scores_per_class[category]:
                            f.write(
                                f"- {metric}: {scores_per_class[category][metric]:.4f}\n"
                            )

                    f.write("Per-class (top {}):\n".format(self.top_k))
                    for metric in ["recall", "precision", "f1"]:
                        if metric in scores_per_class_topk[category]:
                            f.write(
                                f"- {metric}: {scores_per_class_topk[category][metric]:.4f}\n"
                            )

                    f.write("Overall (threshold {:.2f}):\n".format(self.threshold))
                    for metric in ["recall", "precision", "f1", "bacc"]:
                        if metric in scores_overall[category]:
                            f.write(
                                f"- {metric}: {scores_overall[category][metric]:.4f}\n"
                            )

                    f.write("Overall (top {}):\n".format(self.top_k))
                    for metric in ["recall", "precision", "f1"]:
                        if metric in scores_overall_topk[category]:
                            f.write(
                                f"- {metric}: {scores_overall_topk[category][metric]:.4f}\n"
                            )

                    # if category in self.attribute_parent_type.keys():
                    #     self.plot_pr_curves(os.path.dirname(output_file), category)
                    if category in self.attribute_parent_type.keys():
                        self.plot_pr_curves_types(
                            os.path.dirname(output_file), category
                        )

            # For overall, we're interested in F1.
            # For per-class, we're interested in mean AP, mean recall, mean balanced accuracy.
            with open(output_file, "a") as f:
                f.write(
                    "| {:<50}\t| AP\t| Recall@K ({})\t| B.Accuracy\t| N_Pos\t| N_Neg\t| rand Pre.\t|\n".format(
                        "Name", self.top_k
                    )
                )
                f.write("-" * 120 + "\n")
                for i_class in range(self.n_class):
                    att = self.idx2attr[i_class]
                    scores_cls_thres = self.get_score_class(i_class)
                    scores_cls_top_k = self.get_score_class(
                        i_class, threshold_type="topk"
                    )
                    f.write(
                        "| {:<50}\t| {:.3f}\t| {:.4f}\t| {:.4f}\t| {:<6}| {:<6}| {:.4f}\t|\n".format(
                            att[:48],
                            scores_cls_thres.ap,
                            scores_cls_top_k.get_recall(),
                            scores_cls_thres.get_bacc(),
                            scores_cls_thres.n_pos,
                            scores_cls_thres.n_neg,
                            scores_cls_thres.rand_precision,
                        )
                    )

        # Flattens dictionary
        results = pd.json_normalize(results, sep="/")
        results = results.to_dict(orient="records")[0]

        return results


class GroupClassMetric(object):
    def __init__(self, metric_type):
        """This class computes all metrics for a group of attributes.

        Args:
        - metric_type: 'overall' or 'per-class'.
        """
        self.metric_type = metric_type

        if metric_type == "overall":
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
                for name in [
                    "recall",
                    "tnr",
                    "acc",
                    "bacc",
                    "precision",
                    "f1",
                    "ap",
                    "rand_precision",
                ]
            }

    def add_class(self, class_metric):
        """Adds computed metrics of a class into this group."""
        if self.metric_type == "overall":
            self.true_pos += class_metric.true_pos
            self.false_pos += class_metric.false_pos
            self.true_neg += class_metric.true_neg
            self.false_neg += class_metric.false_neg
            self.n_pos += class_metric.n_pos
            self.n_neg += class_metric.n_neg
        else:
            self.metric["recall"].append(class_metric.get_recall())
            self.metric["tnr"].append(class_metric.get_tnr())
            self.metric["acc"].append(class_metric.get_acc())
            self.metric["bacc"].append(class_metric.get_bacc())
            self.metric["precision"].append(class_metric.get_precision())
            self.metric["f1"].append(class_metric.get_f1())
            self.metric["ap"].append(class_metric.ap)
            self.metric["rand_precision"].append(class_metric.rand_precision)

    def get_recall(self):
        """Computes recall."""
        if self.metric_type == "overall":
            n_pos_pred = self.true_pos + self.false_pos
            if n_pos_pred == 0 and self.n_pos > 0:
                # Model makes 0 positive prediction.
                # This is a special case: we fall back to precision = 1 and recall = 0.
                return 0
            if self.n_pos > 0:
                return self.true_pos / self.n_pos
            return -1
        else:
            if -1 not in self.metric["recall"]:
                # return np.mean(self.metric['recall'])
                return np.nanmean(self.metric["recall"])
            else:
                # exclude classes where the are no gt pos instances (e.t. recall=-1)
                valid_metric = [m for m in self.metric["recall"] if m != -1]
                if len(valid_metric) > 0:
                    return np.nanmean(valid_metric)
            return -1

    def get_tnr(self):
        """Computes true negative rate."""
        if self.metric_type == "overall":
            if self.n_neg > 0:
                return self.true_neg / self.n_neg
            return -1
        else:
            if -1 not in self.metric["tnr"]:
                # return np.mean(self.metric['tnr'])
                return np.nanmean(self.metric["tnr"])
            else:
                # exclude classes where the are no gt neg instances (e.t. tnr=-1)
                valid_metric = [m for m in self.metric["tnr"] if m != -1]
                if len(valid_metric) > 0:
                    return np.nanmean(valid_metric)
            return -1

    def get_acc(self):
        """Computes accuracy."""
        if self.metric_type == "overall":
            if self.n_pos + self.n_neg > 0:
                return (self.true_pos + self.true_neg) / (self.n_pos + self.n_neg)
            return -1
        else:
            if -1 not in self.metric["acc"]:
                # return np.mean(self.metric['acc'])
                return np.nanmean(self.metric["acc"])
            else:
                # exclude classes where the are no gt (pos and neg) instances (e.t. acc=-1)
                valid_metric = [m for m in self.metric["acc"] if m != -1]
                if len(valid_metric) > 0:
                    return np.nanmean(valid_metric)
            return -1

    def get_bacc(self):
        """Computes balanced accuracy."""
        if self.metric_type == "overall":
            recall = self.get_recall()
            tnr = self.get_tnr()
            if recall == -1 or tnr == -1:
                return -1
            return (recall + tnr) / 2.0
        else:
            if -1 not in self.metric["bacc"]:
                # return np.mean(self.metric['bacc'])
                return np.nanmean(self.metric["bacc"])
            else:
                # exclude classes where the are no gt (pos or neg) instances (e.t. bacc=-1)
                valid_metric = [m for m in self.metric["bacc"] if m != -1]
                if len(valid_metric) > 0:
                    return np.nanmean(valid_metric)
            return -1

    def get_precision(self):
        """Computes precision."""
        if self.metric_type == "overall":
            n_pos_pred = self.true_pos + self.false_pos
            if n_pos_pred == 0:
                # Model makes 0 positive prediction.
                # This is a special case: we fall back to precision = 1 and recall = 0.
                return 1
            return self.true_pos / n_pos_pred
        else:
            if -1 not in self.metric["precision"]:
                # return np.mean(self.metric['precision'])
                return np.nanmean(self.metric["precision"])
            return -1

    def get_f1(self):
        """Computes F1."""
        if self.metric_type == "overall":
            recall = self.get_recall()
            precision = self.get_precision()
            if precision + recall > 0:
                return 2 * precision * recall / (precision + recall)
            return 0
        else:
            if -1 not in self.metric["f1"]:
                # return np.mean(self.metric['f1'])
                # return np.nanmean(self.metric["f1"])
                f1_metric = self.metric["f1"]
                return -1 if np.isnan(f1_metric).all() else np.nanmean(f1_metric)
            return -1

    def get_ap(self):
        """Computes mAP."""
        assert self.metric_type == "per-class"
        # return np.mean(self.metric['ap'])
        # return np.nanmean(self.metric["ap"])
        ap_metric = np.asarray(self.metric["ap"])
        return np.NaN if np.isnan(ap_metric).all() else np.nanmean(ap_metric)

    def get_random_precision(self):
        """Computes random precision."""
        if self.metric_type == "overall":
            rand_true_pos = self.n_pos / 2
            rand_false_pos = self.n_neg / 2
            rand_pos_pred = rand_true_pos + rand_false_pos
            if rand_pos_pred == 0:
                # Model makes 0 positive prediction.
                # This is a special case: we fall back to precision = 1 and recall = 0.
                return 1
            return rand_true_pos / rand_pos_pred
        else:
            if -1 not in self.metric["rand_precision"]:
                # return np.mean(self.metric['precision'])
                return np.nanmean(self.metric["rand_precision"])
            return -1


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
            self.pr_curve = {"p": [], "r": [], "t": []}
            self.rand_precision = 1
            return

        self.true_pos = ((gt_label == 1) & (pred == 1)).sum()
        self.false_pos = ((gt_label == 0) & (pred == 1)).sum()
        self.true_neg = ((gt_label == 0) & (pred == 0)).sum()
        self.false_neg = ((gt_label == 1) & (pred == 0)).sum()

        # Number of groundtruth positives & negatives.
        self.n_pos = self.true_pos + self.false_neg
        self.n_neg = self.false_pos + self.true_neg

        # Calculate random precision
        rand_true_pos = self.n_pos / 2
        rand_false_pos = self.n_neg / 2
        # rand_true_neg = self.n_neg/2
        # rand_false_neg = self.n_pos/2
        rand_pos_pred = rand_true_pos + rand_false_pos
        if rand_pos_pred == 0:
            # Model makes 0 positive prediction.
            # This is a special case: we fall back to precision = 1 and recall = 0.
            self.rand_precision = 1
        else:
            self.rand_precision = rand_true_pos / rand_pos_pred

        # AP score.
        self.ap = -1
        self.pr_curve = {"p": [], "r": [], "t": []}

    def get_recall(self):
        """Computes recall."""
        n_pos_pred = self.true_pos + self.false_pos
        if n_pos_pred == 0 and self.n_pos > 0:
            # Model makes 0 positive prediction.
            # This is a special case: we fall back to precision = 1 and recall = 0.
            return 0
        if self.n_pos > 0:
            return self.true_pos / self.n_pos
        return -1

    def get_tnr(self):
        """Computes true negative rate."""
        if self.n_neg > 0:
            return self.true_neg / self.n_neg
        return -1

    def get_acc(self):
        """Computes accuracy."""
        if self.n_pos + self.n_neg > 0:
            return (self.true_pos + self.true_neg) / (self.n_pos + self.n_neg)
        return -1

    def get_bacc(self):
        """Computes balanced accuracy."""
        recall = self.get_recall()
        tnr = self.get_tnr()
        if recall == -1 or tnr == -1:
            return -1
        return (recall + tnr) / 2.0

    def get_precision(self):
        """Computes precision."""
        n_pos_pred = self.true_pos + self.false_pos
        if n_pos_pred == 0:
            # Model makes 0 positive prediction.
            # This is a special case: we fall back to precision = 1 and recall = 0.
            return 1
        return self.true_pos / n_pos_pred

    def get_f1(self):
        """Computes F1."""
        recall = self.get_recall()
        precision = self.get_precision()

        if precision + recall > 0:
            return 2 * precision * recall / (precision + recall)
        return float("nan")


def print_metric_table(att_evaluator, results):
    # print(results)
    all_metrics = [
        "PCrand_P",
        "PC_ap",
    ]
    att_type_idx = {}
    table_results = {
        "Type": [],
    }
    metrics = []
    for m in all_metrics:
        m_name = m.format(thr=att_evaluator.threshold, top_k=att_evaluator.top_k)
        metrics.append(m_name)
        table_results[m_name] = []

    print("Printing metrics")
    return_metrics = {}
    for metric_des, score in results.items():
        if len(metric_des.split("/")) == 3:
            metric_type, att_type, metric = metric_des.split("/")
        elif len(metric_des.split("/")) == 4:
            metric_type, _, att_type, metric = metric_des.split("/")
        metric_type = metric_type.replace("_", "")
        metric = metric.replace("precision", "P").replace("recall", "R")
        metric_name = metric_type + "_" + metric
        if "rand" in metric_name and "PCt" in metric_name:
            metric_name = "PCrand_" + metric_name.split("_")[-1]
        if "PCt" in metric_name and "_ap" in metric_name:
            metric_name = "PC_ap"

        if metric_name not in table_results.keys():
            continue

        if att_type not in table_results["Type"]:
            att_type_idx[att_type] = len(table_results["Type"])
            table_results["Type"].append(att_type)

        assert len(table_results[metric_name]) == att_type_idx[att_type]

        table_results[metric_name].append(score)

        if "ap" in metric_name:
            return_metrics[metric_name + "/" + att_type] = score

    table = tabulate(
        table_results,
        headers="keys",
        tablefmt="pipe",
        floatfmt=".3f",
        numalign="left",
    )
    # print("Per-att_group: \n" + table)
    att_evaluator.logger.info("Per-att_group: \n" + table)

    return return_metrics


if __name__ == "__main__":
    from ovad import OVAD

    # Load the labels
    dataset_name = "ovad1200_licence"
    gt_ovad_api = OVAD(dataset_name + ".json")

    logger = logging.getLogger(__name__)
    # Make the dictionaries for evaluator
    attr2idx = {}
    attr_type = {}
    attr_parent_type = {}
    attribute_head_tail = {"head": set(), "medium": set(), "tail": set()}

    for att in gt_ovad_api.atts.values():
        attr2idx[att["name"]] = att["id"]

        if att["type"] not in attr_type.keys():
            attr_type[att["type"]] = set()
        attr_type[att["type"]].add(att["name"])

        if att["parent_type"] not in attr_parent_type.keys():
            attr_parent_type[att["parent_type"]] = set()
        attr_parent_type[att["parent_type"]].add(att["type"])

        attribute_head_tail[att["freq_set"]].add(att["name"])

    attr_type = {key: list(val) for key, val in attr_type.items()}
    attr_parent_type = {key: list(val) for key, val in attr_parent_type.items()}
    attribute_head_tail = {key: list(val) for key, val in attribute_head_tail.items()}

    for key, values in attribute_head_tail.items():
        print(key, len(values))

    overall_metrics = {}

    # Set the ground truth labels
    ground_truth_labels = []
    for ann in gt_ovad_api.anns.values():
        ground_truth_labels.append(np.asarray(ann["att_vec"]))
    ground_truth_labels = np.asarray(ground_truth_labels)
    # ignore value in evaluator is 2
    ground_truth_labels[ground_truth_labels == -1] = 2

    # Build evaluator
    evaluator = AttEvaluator(
        attr2idx,
        attr_type=attr_type,
        attr_parent_type=attr_parent_type,
        attr_headtail=attribute_head_tail,
        att_seen_unseen={},
        dataset_name=dataset_name,
        threshold=0.5,
        top_k=8,
        exclude_atts=[],
    )

    # Set the predictions
    random_predictions = np.random.rand(len(ground_truth_labels), len(attr2idx)).astype(
        "float"
    )

    # Run evaluation
    output_file_fun = os.path.join("output", "{}_random.log".format(dataset_name))
    results = evaluator.print_evaluation(
        pred=random_predictions.copy(),
        gt_label=ground_truth_labels.copy(),
        output_file=output_file_fun,
    )
    # Print results in table
    print_metric_table(evaluator, results)
