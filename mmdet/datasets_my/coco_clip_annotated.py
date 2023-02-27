import contextlib
import io
import itertools
import logging
import os.path
import os.path as osp
import pickle
import tempfile
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from mmdet.utils.api_wrappers import COCO, COCOeval
from ..datasets.builder import DATASETS
from ..datasets.custom import CustomDataset
from ..datasets.pipelines import Compose
from .evaluate_tools.evaluate_attributes import Evaluator
import re
from string import punctuation


@DATASETS.register_module()
class CocoCLIPAnnDataset(CustomDataset):
    def __init__(self,
                 attributes_file,
                 annotations_file,
                 pipeline,
                 attribute_id_map=None,
                 img_prefix='',
                 att_split='val2014',
                 test_mode=False,
                 file_client_args=dict(backend='disk')
                 ):
        self.file_client = mmcv.FileClient(**file_client_args)

        self.attributes_dataset = pickle.load(open(attributes_file, 'rb'))
        self.coco = COCO(annotations_file)
        self.img_prefix = img_prefix
        self.test_mode = test_mode

        self.pipeline = Compose(pipeline)

        self.patch_ids = []
        split = att_split
        # get all attribute vectors for this split
        for patch_id in self.attributes_dataset['ann_vecs'].keys():
            if self.attributes_dataset['split'][patch_id] == split:
                self.patch_ids.append(patch_id)

        # self.patch_ids = self.patch_ids[:16*8]
        # list of attribute names
        self.attributes = sorted(
            self.attributes_dataset['attributes'], key=lambda x: x['id'])
        self.attribute_id_map = mmcv.load(attribute_id_map)

    def __len__(self):
        return len(self.patch_ids)

    def _rand_another(self):
        return np.random.randint(len(self))

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_train_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another()
                continue
            return data

    def prepare_train_img(self, index):
        patch_id = self.patch_ids[index]

        attrs = self.attributes_dataset['ann_vecs'][patch_id]
        attrs = (attrs >= 0.5).astype(np.float)

        ann_id = self.attributes_dataset['patch_id_to_ann_id'][patch_id]
        # coco.loadImgs returns a list
        ann = self.coco.load_anns(ann_id)[0]
        img_info = self.coco.load_imgs(ann['image_id'])[0]


        x1, y1, w, h = ann['bbox']
        inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
        inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
        if not self.test_mode:
            if inter_w * inter_h == 0:
                print('box is too small')
                return None
            if ann['area'] <= 0 or w < 1 or h < 1:
                print('box is too small')
                return None
        bbox = [x1, y1, x1 + w, y1 + h]

        gt_bboxes = np.array(bbox, dtype=np.float32)

        results = dict(img_info=img_info, ann_info=ann, attrs=attrs)
        results['img_prefix'] = self.img_prefix
        results['img_info']['filename'] = img_info['file_name']
        results['ann_info']['bboxes'] = np.array(gt_bboxes).reshape(1, 4)
        results['bbox_fields'] = []
        results = self.pipeline(results)
        # results['gt_bboxes'] = results['gt_bboxes']
        return results

    def get_labels(self):
        gt_labels = []
        for patch_id in self.patch_ids:
            attrs = self.attributes_dataset['ann_vecs'][patch_id]
            attrs = (attrs >= 0.5).astype(np.float)
            gt_labels.append(attrs)
        return np.stack(gt_labels, axis=0)

    def evaluate(self,
                 results,
                 **kwargs
                 ):
        results = np.array(results)
        # print()
        # print(results[:, 0])
        preds = torch.from_numpy(results)
        gts = self.get_labels()
        gts = torch.from_numpy(gts)
        pred_prob = preds.sigmoid()

        pred_att = pred_prob > 0.5

        t_p_samples = torch.sum(pred_att * gts)
        p_samples = torch.sum(gts)

        result = {
            "recall": t_p_samples / p_samples,
            'precision': t_p_samples / torch.sum(pred_att)
        }

        evaluator = Evaluator(self.attribute_id_map)
        gt_label = gts.data.cpu().numpy()  # Nx620
        pred_prob = pred_prob.data.cpu().numpy()
        scores_overall, scores_per_class = evaluator.evaluate(pred_prob, gt_label)

        scores_overall_topk, scores_per_class_topk = evaluator.evaluate(pred_prob, gt_label, threshold_type='topk')

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


        return result