import json
import logging
import os
import os.path as osp
import pickle
import random
import tempfile
import warnings
from collections import OrderedDict, defaultdict

import mmcv
import numpy as np
import torch

from ..datasets.builder import DATASETS
from torch.utils.data import Dataset
from ..datasets.pipelines import Compose
from .evaluate_tools import cal_metrics
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps


@DATASETS.register_module()
class VAWProposalDataset(Dataset):

    CLASSES = None

    def __init__(self,
                 data_root,
                 pipeline,
                 pattern,
                 test_mode=False,
                 file_client_args=dict(backend='disk')
                 ):
        assert pattern in ['train', 'val', 'test']
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.data_root = data_root
        if pattern == 'train':
            self.instances, self.img_instances_pair = self.read_data(["train_part1.json", "train_part2.json"])
        elif pattern == 'val':
            self.instances, self.img_instances_pair = self.read_data(['val.json'])
        elif pattern == 'test':
            self.instances, self.img_instances_pair = self.read_data(['test.json'])
        print('num instances: ', len(self.instances))
        print('data len: ', len(self.img_instances_pair))
        self.error_list = set()
        self.img_ids = list(self.img_instances_pair.keys())

        self.instances = self.get_instances()
        text_file = os.path.join(self.data_root, "VAW/attribute_index.json")
        self.classname_maps = json.load(open(text_file))

    def get_instances(self):

        proposals = json.load(open('/data/kyanchen/prompt1/tools/results/EXP20220628_0/FasterRCNN_R50_OpenImages.proposal.json', 'r'))
        img_proposal_pair = {}
        for instance in proposals:
            img_id = instance['image_id']
            img_proposal_pair[img_id] = img_proposal_pair.get(img_id, []) + [instance]

        instances = []
        for img_id in self.img_ids:
            gt_bboxes = [instance['instance_bbox'] for instance in self.img_instances_pair[img_id]]
            gt_bboxes = np.array(gt_bboxes).reshape(-1, 4)
            gt_bboxes[:, 2:] = gt_bboxes[:, :2] + gt_bboxes[:, 2:]
            for proposal in img_proposal_pair[img_id]:
                if proposal['score'] < 0.55:
                    continue
                box = np.array(proposal['bbox']).reshape(-1, 4)
                iou = bbox_overlaps(box, gt_bboxes)[0]
                box_ind = np.argmax(iou)
                if iou[box_ind] < 0.6:
                    continue
                # import pdb
                # pdb.set_trace()
                instance = self.img_instances_pair[img_id][box_ind]
                instance['instance_bbox'] = box[0].tolist()
                instances.append(instance)

        return instances


    def read_data(self, json_file_list):
        json_data = [json.load(open(self.data_root + '/VAW/' + x)) for x in json_file_list]
        instances = []
        [instances.extend(x) for x in json_data]
        img_instances_pair = {}
        for instance in instances:
            img_id = instance['image_id']
            img_instances_pair[img_id] = img_instances_pair.get(img_id, []) + [instance]
        # sub = {}
        # sub_keys = list(img_instances_pair.keys())[:10]
        # for k in sub_keys:
        #     sub[k] = img_instances_pair[k]

        return instances, img_instances_pair

    def __len__(self):
        return len(self.instances)

    def test_proposal_atts(self, idx):
        results = self.instances[idx]
        results['img_prefix'] = os.path.abspath(self.data_root) + '/VG/VG_100K'
        results['img_info'] = {}
        results['img_info']['filename'] = f'{results["image_id"]}.jpg'
        results['instance_bbox'] = results["instance_bbox"]
        results = self.pipeline(results)
        return results

    def __getitem__(self, idx):
        if self.test_mode:
            return self.test_proposal_atts(idx)
        if idx in self.error_list and not self.test_mode:
            idx = np.random.randint(0, len(self))
        item = self.data[idx]
        results = item.__dict__.copy()
        results['img_prefix'] = os.path.abspath(self.data_root) + '/VG/VG_100K'
        results['img_info'] = {}
        results['img_info']['filename'] = f'{item.image_id}.jpg'
        results['instance_bbox'] = item.instance_bbox
        results['gt_labels'] = item.label.astype(np.int)
        if self.test_mode:
            results = self.pipeline(results)
        else:
            try:
                # print(results)
                results = self.pipeline(results)
            except Exception as e:
                print(e)
                self.error_list.add(idx)
                self.error_list.add(results['img_info']['filename'])
                print(self.error_list)
                if not self.test_mode:
                    results = self.__getitem__(np.random.randint(0, len(self)))
        return results

    def get_labels(self):
        np_gt_labels = []
        for results in self.instances:
            positive_attributes = results['positive_attributes']
            negative_attributes = results['negative_attributes']
            label = np.ones(len(self.classname_maps.keys())) * 2
            for att in positive_attributes:
                label[self.classname_maps[att]] = 1
            for att in negative_attributes:
                label[self.classname_maps[att]] = 0

            gt_labels = label.astype(np.int)

            np_gt_labels.append(gt_labels.astype(np.int))
        return np.stack(np_gt_labels, axis=0)

    def evaluate(self,
                 results,
                 metric='mAP',
                 per_class_out_file=None,
                 is_logit=True
                 ):

        results = np.array(results)
        preds = torch.from_numpy(results)
        gts = self.get_labels()
        gts = torch.from_numpy(gts)

        output = cal_metrics(self.data_root + '/VAW',
                             preds, gts,
                             return_all=True,
                             return_evaluator=per_class_out_file,
                             is_logit=is_logit)

        if per_class_out_file:
            scores_overall, scores_per_class, scores_overall_topk, scores_per_class_topk, evaluator = output
        else:
            scores_overall, scores_per_class, scores_overall_topk, scores_per_class_topk = output

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

            if per_class_out_file:
                mmcv.mkdir_or_exist(osp.basename(per_class_out_file))
                with open(per_class_out_file, 'w') as f:
                    f.write('| {:<18}| AP\t\t| Recall@K\t| B.Accuracy\t| N_Pos\t| N_Neg\t|\n'.format('Name'))
                    f.write('-----------------------------------------------------------------------------------------------------\n')
                    for i_class in range(evaluator.n_class):
                        att = evaluator.idx2attr[i_class]
                        f.write('| {:<18}| {:.4f}\t| {:.4f}\t| {:.4f}\t\t| {:<6}| {:<6}|\n'.format(
                            att,
                            evaluator.get_score_class(i_class).ap,
                            evaluator.get_score_class(i_class, threshold_type='topk').get_recall(),
                            evaluator.get_score_class(i_class).get_bacc(),
                            evaluator.get_score_class(i_class).n_pos,
                            evaluator.get_score_class(i_class).n_neg))

        return results


#     "image_id": "2373241",
#     "instance_id": "2373241004",
#     "instance_bbox": [0.0, 182.5, 500.16666666666663, 148.5],
#     "instance_polygon": [[[432.5, 214.16666666666669], [425.8333333333333, 194.16666666666666], [447.5, 190.0], [461.6666666666667, 187.5], [464.1666666666667, 182.5], [499.16666666666663, 183.33333333333331], [499.16666666666663, 330.0], [3.3333333333333335, 330.0], [0.0, 253.33333333333334], [43.333333333333336, 245.0], [60.833333333333336, 273.3333333333333], [80.0, 293.3333333333333], [107.5, 307.5], [133.33333333333334, 309.16666666666663], [169.16666666666666, 295.8333333333333], [190.83333333333331, 274.1666666666667], [203.33333333333334, 252.5], [225.0, 260.0], [236.66666666666666, 254.16666666666666], [260.0, 254.16666666666666], [288.3333333333333, 253.33333333333334], [287.5, 257.5], [271.6666666666667, 265.0], [324.1666666666667, 281.6666666666667], [369.16666666666663, 274.1666666666667], [337.5, 261.6666666666667], [338.3333333333333, 257.5], [355.0, 261.6666666666667], [357.5, 257.5], [339.1666666666667, 255.0], [337.5, 240.83333333333334], [348.3333333333333, 238.33333333333334], [359.1666666666667, 248.33333333333331], [377.5, 251.66666666666666], [397.5, 248.33333333333331], [408.3333333333333, 236.66666666666666], [418.3333333333333, 220.83333333333331], [427.5, 217.5], [434.16666666666663, 215.0]]],
#     "object_name": "floor",
#     "positive_attributes": ["tiled", "gray", "light colored"],
#     "negative_attributes": ["multicolored", "maroon", "weathered", "speckled", "carpeted"]
# }

class DataItem:
    def __init__(
        self, image_id, instance_id, instance_bbox,
        object_name, positive_attributes, negative_attributes,
        label
    ):
        self.image_id = image_id
        self.instance_id = instance_id
        self.instance_bbox = instance_bbox
        self.object_name = object_name
        self.positive_attributes = positive_attributes
        self.negative_attributes = negative_attributes
        self.label = label

    def set_label(self, classname_maps):
        self.label = np.ones(len(classname_maps.keys())) * 2
        for att in self.positive_attributes:
            self.label[classname_maps[att]] = 1
        for att in self.negative_attributes:
            self.label[classname_maps[att]] = 0
