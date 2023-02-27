import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix
from .metrics import cal_metrics


class AttCls:
    """Evaluator for attributes classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self.prefix_path = f'{self.cfg.DATASET.ROOT}/VAW/'
        self.preds = []
        self.gts = []

    def reset(self):
        self.preds = []
        self.gts = []

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch, num_classes]
        self.preds += [mo]
        self.gts += [gt]

        # pred = mo.max(1)[1]
        # matches = pred.eq(gt).float()
        # self._correct += int(matches.sum().item())
        # self._total += gt.shape[0]

        # self._y_true.extend(gt.data.cpu().numpy().tolist())
        # self._y_pred.extend(pred.data.cpu().numpy().tolist())

        # if self._per_class_res is not None:
        #     for i, label in enumerate(gt):
        #         label = label.item()
        #         matches_i = int(matches[i].item())
        #         self._per_class_res[label].append(matches_i)

    def evaluate(self):
        preds = torch.cat(self.preds, dim=0)
        gts = torch.cat(self.gts, dim=0)
        
        scores_overall, scores_per_class, scores_overall_topk, scores_per_class_topk = cal_metrics(
            self.prefix_path, preds, gts, return_all=True
            )
        
        # CATEGORIES = ['all', 'head', 'medium', 'tail'] + \
        # list(evaluator.attribute_parent_type.keys())
        results = OrderedDict()
        CATEGORIES = ['all']

        for category in CATEGORIES:
            print(f"----------{category.upper()}----------")
            print(f"mAP: {scores_per_class[category]['ap']:.4f}")
            results['all_mAP'] = scores_per_class['all']['ap']
            
            print("Per-class (threshold 0.5):")
            for metric in ['recall', 'precision', 'f1', 'bacc']:
                if metric in scores_per_class[category]:
                    print(f"- {metric}: {scores_per_class[category][metric]:.4f}")
            
            print("Per-class (top 15):")
            for metric in ['recall', 'precision', 'f1']:
                if metric in scores_per_class_topk[category]:
                    print(f"- {metric}: {scores_per_class_topk[category][metric]:.4f}")
        
            print("Overall (threshold 0.5):")
            for metric in ['recall', 'precision', 'f1', 'bacc']:
                if metric in scores_overall[category]:
                    print(f"- {metric}: {scores_overall[category][metric]:.4f}")
            print("Overall (top 15):")
            for metric in ['recall', 'precision', 'f1']:
                if metric in scores_overall_topk[category]:
                    print(f"- {metric}: {scores_overall_topk[category][metric]:.4f}")

        # with open(output, 'w') as f:
        #     f.write('| {:<18}| AP\t\t| Recall@K\t| B.Accuracy\t| N_Pos\t| N_Neg\t|\n'.format('Name'))
        #     f.write('-----------------------------------------------------------------------------------------------------\n')
        #     for i_class in range(evaluator.n_class):
        #         att = evaluator.idx2attr[i_class]
        #         f.write('| {:<18}| {:.4f}\t| {:.4f}\t| {:.4f}\t\t| {:<6}| {:<6}|\n'.format(
        #             att,
        #             evaluator.get_score_class(i_class).ap,
        #             evaluator.get_score_class(i_class, threshold_type='topk').get_recall(),
        #             evaluator.get_score_class(i_class).get_bacc(),
        #             evaluator.get_score_class(i_class).n_pos,
        #             evaluator.get_score_class(i_class).n_neg))
        
        return results

        # results = OrderedDict()
        # acc = 100.0 * self._correct / self._total
        # err = 100.0 - acc
        # macro_f1 = 100.0 * f1_score(
        #     self._y_true,
        #     self._y_pred,
        #     average="macro",
        #     labels=np.unique(self._y_true)
        # )

        # # The first value will be returned by trainer.test()
        # results["accuracy"] = acc
        # results["error_rate"] = err
        # results["macro_f1"] = macro_f1

        # print(
        #     "=> result\n"
        #     f"* total: {self._total:,}\n"
        #     f"* correct: {self._correct:,}\n"
        #     f"* accuracy: {acc:.2f}%\n"
        #     f"* error: {err:.2f}%\n"
        #     f"* macro_f1: {macro_f1:.2f}%"
        # )

        # if self._per_class_res is not None:
        #     labels = list(self._per_class_res.keys())
        #     labels.sort()

        #     print("=> per-class result")
        #     accs = []

        #     for label in labels:
        #         classname = self._lab2cname[label]
        #         res = self._per_class_res[label]
        #         correct = sum(res)
        #         total = len(res)
        #         acc = 100.0 * correct / total
        #         accs.append(acc)
        #         print(
        #             "* class: {} ({})\t"
        #             "total: {:,}\t"
        #             "correct: {:,}\t"
        #             "acc: {:.2f}%".format(
        #                 label, classname, total, correct, acc
        #             )
        #         )
        #     mean_acc = np.mean(accs)
        #     print("* average: {:.2f}%".format(mean_acc))

        #     results["perclass_accuracy"] = mean_acc

        # if self.cfg.TEST.COMPUTE_CMAT:
        #     cmat = confusion_matrix(
        #         self._y_true, self._y_pred, normalize="true"
        #     )
        #     save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
        #     torch.save(cmat, save_path)
        #     print('Confusion matrix is saved to "{}"'.format(save_path))
