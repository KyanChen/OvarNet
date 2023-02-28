import contextlib
import copy
import io
import itertools
import json
import logging
import math
import os
import os.path as osp
import pickle
import random
import tempfile
import warnings

from pycocotools.coco import COCO
from terminaltables import AsciiTable
from mmcv.ops import batched_nms
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from collections import OrderedDict, defaultdict
import imagesize
import cv2
import mmcv
import numpy as np
import torch
from mmcv import tensor2imgs, print_log
from mmcv.parallel import DataContainer
from pycocotools.cocoeval import COCOeval
from sklearn import metrics
from torchmetrics.detection import MeanAveragePrecision

# from .evaluate_tools.custom_coco_eval import CustomCOCOEvaluator
from ..core import eval_recalls, eval_map, bbox2result
from ..datasets.builder import DATASETS
from torch.utils.data import Dataset
from ..datasets.pipelines import Compose
from .evaluate_tools import cal_metrics
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps


attribute_index_file = dict(
    att_file='attributes/VAW/base2novel_att2id.json',
    att_group='base+novel',
)

@DATASETS.register_module()
class InferenceRPNAttributeDataset(Dataset):

    CLASSES = None

    def __init__(self,
                 data_root='/data/kyanchen/prompt/data',
                 pipeline=None,
                 dataset_split='test',
                 attribute_index_file=attribute_index_file,
                 dataset_names=['vaw'],
                 dataset_balance=False,
                 kd_pipeline=None,
                 test_mode=True,
                 test_content='box_free',
                 mult_proposal_score=False,
                 file_client_args=dict(backend='disk')
                 ):
        super(InferenceRPNAttributeDataset, self).__init__()
        assert dataset_split in ['train', 'val', 'test']
        self.dataset_split = dataset_split
        self.test_mode = test_mode
        self.mult_proposal_score = mult_proposal_score
        # self.pipeline = Compose(pipeline)
        self.data_root = data_root
        self.dataset_names = dataset_names

        self.attribute_index_file = attribute_index_file
        self.att2id = {}
        self.att_seen_unseen = {}
        if 'att_file' in attribute_index_file.keys():
            file = attribute_index_file['att_file']
            att2id = json.load(open(file, 'r'))
            att_group = attribute_index_file['att_group']
            if att_group in ['common1', 'common2']:
                self.att2id = att2id[att_group]
                self.att_seen_unseen['seen'] = list(att2id['common1'].keys())
                self.att_seen_unseen['unseen'] = list(att2id['common2'].keys())
            elif att_group in ['common', 'rare']:
                self.att2id = att2id[att_group]
                self.att_seen_unseen['seen'] = list(att2id['common'].keys())
                self.att_seen_unseen['unseen'] = list(att2id['rare'].keys())
            elif att_group == 'common1+common2':
                self.att2id.update(att2id['common1'])
                self.att2id.update(att2id['common2'])
                self.att_seen_unseen['seen'] = list(att2id['common1'].keys())
                self.att_seen_unseen['unseen'] = list(att2id['common2'].keys())
            elif att_group == 'common+rare':
                self.att2id.update(att2id['common'])
                self.att2id.update(att2id['rare'])
                self.att_seen_unseen['seen'] = list(att2id['common'].keys())
                self.att_seen_unseen['unseen'] = list(att2id['rare'].keys())
            elif att_group == 'base+novel':
                self.att2id.update(att2id['base'])
                self.att2id.update(att2id['novel'])
                self.att_seen_unseen['seen'] = list(att2id['base'].keys())
                self.att_seen_unseen['unseen'] = list(att2id['novel'].keys())
            else:
                raise NameError
        self.category2id = {}
        self.category_seen_unseen = {}
        if 'category_file' in attribute_index_file.keys():
            file = attribute_index_file['category_file']
            category2id = json.load(open(file, 'r'))
            att_group = attribute_index_file['category_group']
            if att_group in ['common1', 'common2', 'common', 'rare']:
                self.category2id = category2id[att_group]
                self.category_seen_unseen['seen'] = list(category2id['common1'].keys())
                self.category_seen_unseen['unseen'] = list(category2id['common2'].keys())
            elif att_group == 'common1+common2':
                self.category2id.update(category2id['common1'])
                self.category2id.update(category2id['common2'])
                self.category_seen_unseen['seen'] = list(category2id['common1'].keys())
                self.category_seen_unseen['unseen'] = list(category2id['common2'].keys())
            elif att_group == 'common+rare':
                self.category2id.update(category2id['common'])
                self.category2id.update(category2id['rare'])
            else:
                raise NameError
        self.att2id = {k: v - min(self.att2id.values()) for k, v in self.att2id.items()}
        self.category2id = {k: v - min(self.category2id.values()) for k, v in self.category2id.items()}

        self.id2images = {}
        self.id2instances = {}

        if 'coco' in self.dataset_names:
            id2images_coco, id2instances_coco = self.read_data_coco(dataset_split)
            self.id2images.update(id2images_coco)
            self.id2instances.update(id2instances_coco)
            self.id2instances.pop('coco_200365', None)
            self.id2instances.pop('coco_183338', None)
            self.id2instances.pop('coco_550395', None)
            self.id2instances.pop('coco_77039', None)
            self.id2instances.pop('coco_340038', None)
            # self.id2instances.pop('coco_147195', None)
            # self.id2instances.pop('coco_247306', None)
            self.id2instances.pop('coco_438629', None)
            self.id2instances.pop('coco_284932', None)

        if 'vaw' in self.dataset_names:
            assert dataset_split in ['train', 'test']
            id2images_vaw, id2instances_vaw = self.read_data_vaw(dataset_split)
            self.id2images.update(id2images_vaw)
            self.id2instances.update(id2instances_vaw)
            # self.id2instances.pop('vaw_713545', None)
            # self.id2instances.pop('vaw_2369080', None)

        if 'ovadcate' in self.dataset_names:
            if dataset_split == 'test':
                dataset_split == 'val'
            id2images_ovad, id2instances_ovad = self.read_data_ovad('cate')
            self.id2images.update(id2images_ovad)
            self.id2instances.update(id2instances_ovad)

        if 'ovadattr' in self.dataset_names:
            if dataset_split == 'test':
                dataset_split == 'val'
            id2images_ovad, id2instances_ovad = self.read_data_ovad('attr')
            self.id2images.update(id2images_ovad)
            self.id2instances.update(id2instances_ovad)

        if 'ovadgen' in self.dataset_names:
            id2images_ovadgen, id2instances_ovadgen = self.read_data_ovadgen(dataset_split)
            self.id2images.update(id2images_ovadgen)
            self.id2instances.update(id2instances_ovadgen)

        if not self.test_mode:
            # filter images too small and containing no annotations
            self.img_ids = self._filter_imgs()
            self._set_group_flag()
        else:
            self.img_ids = list(self.id2images.keys())
        # [:20] + list(self.id2instances.keys())[-20:]

        img_ids_per_dataset = {}
        for x in self.img_ids:
            img_ids_per_dataset[x.split('_')[0]] = img_ids_per_dataset.get(x.split('_')[0], []) + [x]

        rank, world_size = get_dist_info()
        if rank == 0:
            for k, v in img_ids_per_dataset.items():
                print('dataset: ', k, ' len - ', len(v))
        if dataset_balance and not test_mode:
            balance_frac = round(len(img_ids_per_dataset['coco']) / len(img_ids_per_dataset['vaw']))
            self.img_ids = img_ids_per_dataset['coco'] + balance_frac * img_ids_per_dataset['vaw']
            if rank == 0:
                print('balance dataset fractor = ', balance_frac)
                data_len = {}
                for x in self.img_ids:
                    data_len[x.split('_')[0]] = data_len.get(x.split('_')[0], 0) + 1
                for k, v in data_len.items():
                    print('data len: ', k, ' - ', v)

            flag_dataset = [x.split('_')[0] for x in self.img_ids]
            dataset_types = {'coco': 0, 'vaw': 1}
            flag_dataset = [dataset_types[x] for x in flag_dataset]
            self.flag_dataset = np.array(flag_dataset, dtype=np.int)
        self.error_list = set()
        self.test_content = test_content
        assert self.test_content in ['box_given', 'box_free']

    def read_data_coco(self, pattern):
        if pattern == 'test':
            pattern = 'val'
        json_file = 'instances_train2017' if pattern == 'train' else 'instances_val2017'
        # json_file = 'lvis_v1_train' if pattern == 'train' else 'instances_val2017'
        json_data = json.load(open(self.data_root + f'/COCO/annotations/{json_file}.json', 'r'))
        id2name = {x['id']: x['name'] for x in json_data['categories']}
        id2images = {}
        id2instances = {}
        for data in json_data['images']:
            img_id = 'coco_' + str(data['id'])
            data['file_name'] = f'{data["id"]:012d}.jpg'
            id2images[img_id] = data
        for data in json_data['annotations']:
            img_id = 'coco_' + str(data['image_id'])
            data['name'] = id2name[data['category_id']]
            id2instances[img_id] = id2instances.get(img_id, []) + [data]
        return id2images, id2instances

    def read_data_ovad(self, pattern):
        json_data = json.load(open('../attributes/OVAD/ovad2000.json', 'r'))
        instances = json_data['annotations']
        categoryid2name = {x['id']: x['name'] for x in json_data['categories']}
        attributeid2name = {x['id']: x for x in json_data['attributes']}
        id2images = {}
        id2instances = {}
        for instance in instances:
            img_id = f'ovad{pattern}_' + str(instance['image_id'])
            instance['name'] = categoryid2name[instance['category_id']]
            instance['positive_attributes'] = []
            instance['negative_attributes'] = []
            for idx, att_ann in enumerate(instance['att_vec']):
                '''
                % 1 = positive attribute
                % 0 = negative attribute
                % -1 = ignore attribute
                '''
                if att_ann in [0, 1]:
                    att = attributeid2name[idx]['name']
                    if att_ann == 1:
                        instance['positive_attributes'] += [att]
                    else:
                        instance['negative_attributes'] += [att]
            id2instances[img_id] = id2instances.get(img_id, []) + [instance]

        for data in json_data['images']:
            img_id = f'ovad{pattern}_' + str(data['id'])
            id2images[img_id] = data

        return id2images, id2instances

    def read_data_vaw(self, pattern):
        json_files = ['train_part1', 'train_part2'] if pattern == 'train' else [f'{pattern}']
        json_data = [json.load(open(self.data_root + '/VAW/' + f'{x}.json', 'r')) for x in json_files]
        instances = []
        [instances.extend(x) for x in json_data]
        id2images = {}
        id2instances = {}
        for instance in instances:
            img_id = 'vaw_' + str(instance['image_id'])
            id2instances[img_id] = id2instances.get(img_id, []) + [instance]
        for img_id in id2instances.keys():
            img_info = {'file_name': f'{img_id.split("_")[-1] + ".jpg"}'}
            img_path = os.path.abspath(self.data_root) + '/VG/VG_100K/' + img_info['file_name']
            w, h = imagesize.get(img_path)
            img_info['width'] = w
            img_info['height'] = h
            id2images[img_id] = img_info
        return id2images, id2instances

    def _filter_imgs(self, min_wh_size=32, min_box_wh_size=4):
        valid_img_ids = []
        for img_id, img_info in self.id2images.items():
            if min(img_info['width'], img_info['height']) < min_wh_size:
                continue
            instances = self.id2instances.get(img_id, [])
            instances_tmp = []

            data_set = img_id.split('_')[0]
            key = 'bbox' if data_set == 'coco' else 'instance_bbox'
            for instance in instances:
                x, y, w, h = instance[key]
                if w < min_box_wh_size or h < min_box_wh_size:
                    continue
                if data_set == 'coco':
                    category = instance['name']
                    category_id = self.category2id.get(category, None)  # 未标注的该类别的应该去除
                    if category_id is not None:
                        instances_tmp.append(instance)
                elif data_set == 'vaw':
                    attributes = instance["positive_attributes"] + instance["negative_attributes"]
                    all_attrs = set(self.att2id.keys())  # 未标注的该类别的应该去除
                    if len(set(attributes) & all_attrs):
                        instances_tmp.append(instance)
                elif data_set == 'ovadgen':
                    instances_tmp.append(instance)

            self.id2instances[img_id] = instances_tmp
            if len(instances_tmp) == 0:
                continue
            valid_img_ids.append(img_id)
        return valid_img_ids

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

    def __len__(self):
        return len(self.img_ids)

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for idx, img_id in enumerate(self.img_ids):
            img_info = self.id2images[img_id]
            w, h = img_info['width'], img_info['height']
            if w / h > 1:
                self.flag[idx] = 1

    def get_test_data(self, idx):
        results = self.instances[idx].copy()
        results['img_prefix'] = os.path.abspath(self.data_root) + '/VG/VG_100K'
        results['img_info'] = {}
        results['img_info']['filename'] = f'{results["image_id"]}.jpg'
        x, y, w, h = results["instance_bbox"]
        results['proposals'] = np.array([x, y, x + w, y + h], dtype=np.float32).reshape(1, 4)
        results['bbox_fields'] = ['proposals']
        results = self.pipeline(results)
        return results

    def get_img_instances(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.id2images[img_id]
        instances = self.id2instances[img_id]
        data_set = img_id.split('_')[0]
        if data_set == 'coco':
            data_set_type = 0
            if self.dataset_split == 'test':
                dataset_split = 'val'
            else:
                dataset_split = self.dataset_split
            prefix_path = f'/COCO/{dataset_split}2017'
        elif data_set == 'vaw':
            prefix_path = f'/VG/VG_100K'
            data_set_type = 1
        elif data_set in ['ovadcate', 'ovadattr']:
            if self.dataset_split == 'test':
                dataset_split = 'val'
            else:
                dataset_split = self.dataset_split
            prefix_path = f'/COCO/{dataset_split}2017'
            if data_set == 'ovadcate':
                data_set_type = 0
            elif data_set == 'ovadattr':
                data_set_type = 1
            else:
                raise NameError
        elif data_set == 'ovadgen':
            data_set_type = 1
            prefix_path = f'/ovadgen'
        else:
            raise NameError

        results = {}
        results['data_set_type'] = data_set_type
        results['img_prefix'] = os.path.abspath(self.data_root) + prefix_path
        results['img_info'] = {}
        results['img_info']['filename'] = img_info['file_name']

        bbox_list = []
        attr_label_list = []
        for instance in instances:
            key = 'bbox' if data_set == 'coco' else 'instance_bbox'
            x, y, w, h = instance[key]
            bbox_list.append([x, y, x + w, y + h])

            labels = np.ones(len(self.att2id)+len(self.category2id)) * 2
            labels[len(self.att2id):] = 0
            if data_set == 'vaw' or data_set == 'ovadgen':
                positive_attributes = instance["positive_attributes"]
                negative_attributes = instance["negative_attributes"]
                for att in positive_attributes:
                    if att in self.att_seen_unseen['seen']:
                        att_id = self.att2id.get(att, None)
                        if att_id is not None:
                            labels[att_id] = 1
                for att in negative_attributes:
                    if att in self.att_seen_unseen['seen']:
                        att_id = self.att2id.get(att, None)
                        if att_id is not None:
                            labels[att_id] = 0
            if data_set == 'coco':
                category = instance['name']
                if category in self.category_seen_unseen['seen']:
                    data_set_type = 0
                    category_id = self.category2id.get(category, None)
                    if category_id is not None:
                        labels[category_id+len(self.att2id)] = 1
                else:
                    data_set_type = 4  # 只进行蒸馏损失的计算
            attr_label_list.append(labels)

        gt_bboxes = np.array(bbox_list, dtype=np.float32)
        gt_labels = np.stack(attr_label_list, axis=0).astype(np.int)
        results['gt_bboxes'] = gt_bboxes
        results['bbox_fields'] = ['gt_bboxes']
        results['gt_labels'] = gt_labels
        results['dataset_type'] = data_set_type
        assert len(gt_labels) == len(gt_bboxes)
        if self.kd_pipeline:
            kd_results = results.copy()
            kd_results.pop('gt_labels')
            kd_results.pop('bbox_fields')
        try:
            results = self.pipeline(results)
            if self.kd_pipeline:
                kd_results = self.kd_pipeline(kd_results, 0)
                img_crops = []
                for gt_bboxes in kd_results['gt_bboxes']:
                    kd_results_tmp = kd_results.copy()
                    kd_results_tmp['crop_box'] = gt_bboxes
                    kd_results_tmp = self.kd_pipeline(kd_results_tmp, (1, ':'))
                    img_crops.append(kd_results_tmp['img'])
                img_crops = torch.stack(img_crops, dim=0)
                results['img_crops'] = img_crops

            results['gt_bboxes'] = DataContainer(results['gt_bboxes'], stack=False)
            results['gt_labels'] = DataContainer(results['gt_labels'], stack=False)
            results['img'] = DataContainer(results['img'], padding_value=0, stack=True)
            if self.kd_pipeline:
                results['img_crops'] = DataContainer(results['img_crops'], stack=False)
        except Exception as e:
            print(e)
            self.error_list.add(idx)
            self.error_list.add(img_id)
            print(self.error_list)
            if not self.test_mode:
                results = self.__getitem__(np.random.randint(0, len(self)))
        return results

    def get_test_box_given(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.id2images[img_id]
        instances = self.id2instances[img_id]
        data_set = img_id.split('_')[0]
        if data_set == 'coco':
            data_set_type = 0
            if self.dataset_split == 'test':
                dataset_split = 'val'
            else:
                dataset_split = self.dataset_split
            prefix_path = f'/COCO/{dataset_split}2017'
        elif data_set == 'vaw':
            prefix_path = f'/VG/VG_100K'
            data_set_type = 1
        elif data_set in ['ovadcate', 'ovadattr']:
            if self.dataset_split == 'test':
                dataset_split = 'val'
            else:
                dataset_split = self.dataset_split
            prefix_path = f'/COCO/{dataset_split}2017'
            if data_set == 'ovadcate':
                data_set_type = 0
            elif data_set == 'ovadattr':
                data_set_type = 1
            else:
                raise NameError
        elif data_set == 'ovadgen':
            data_set_type = 1
            prefix_path = f'/ovadgen'
        else:
            raise NameError

        results = {}
        results['img_prefix'] = os.path.abspath(self.data_root) + prefix_path
        results['img_info'] = {}
        results['img_info']['filename'] = img_info['file_name']

        bbox_list = []
        for instance in instances:
            key = 'bbox' if data_set in ['coco', 'label_coco', 'label_cc3m', 'ovadattr', 'ovadcate'] else 'instance_bbox'
            x, y, w, h = instance[key]
            bbox_list.append([x, y, x + w, y + h])

        gt_bboxes = np.array(bbox_list, dtype=np.float32).reshape(-1, 4)
        results['gt_bboxes'] = gt_bboxes
        results['bbox_fields'] = ['gt_bboxes']
        results = self.pipeline(results)

        return results

    def get_test_box_free(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.id2images[img_id]

        data_set = img_id.split('_')[0]
        if data_set == 'coco':
            data_set_type = 0
            if self.dataset_split == 'test':
                dataset_split = 'val'
            else:
                dataset_split = self.dataset_split
            prefix_path = f'/COCO/{dataset_split}2017'
        elif data_set == 'vaw':
            prefix_path = f'/VG/VG_100K'
            data_set_type = 1
        elif data_set in ['ovadcate', 'ovadattr']:
            if self.dataset_split == 'test':
                dataset_split = 'val'
            else:
                dataset_split = self.dataset_split
            prefix_path = f'/COCO/{dataset_split}2017'
            if data_set == 'ovadcate':
                data_set_type = 0
            elif data_set == 'ovadattr':
                data_set_type = 1
            else:
                raise NameError
        elif data_set == 'ovadgen':
            data_set_type = 1
            prefix_path = f'/ovadgen'
        else:
            raise NameError
        results = {}
        results['img_prefix'] = os.path.abspath(self.data_root) + prefix_path
        results['img_info'] = {}
        results['img_info']['filename'] = img_info['file_name']
        results = self.pipeline(results)
        return results

    def __getitem__(self, idx):
        if self.test_mode:
            if self.test_content == 'box_free':
                return self.get_test_box_free(idx)
            elif self.test_content == 'box_given':
                return self.get_test_box_given(idx)
            else:
                assert NotImplementedError
        if idx in self.error_list and not self.test_mode:
            idx = np.random.randint(0, len(self))
        return self.get_img_instances(idx)

        if self.test_mode:
            return self.get_test_data(idx)

        results = self.instances[idx].copy()
        '''
        "image_id": "2373241",
        "instance_id": "2373241004",
        "instance_bbox": [0.0, 182.5, 500.16666666666663, 148.5],
        "instance_polygon": [[[432.5, 214.16666666666669], [425.8333333333333, 194.16666666666666], [447.5, 190.0], [461.6666666666667, 187.5], [464.1666666666667, 182.5], [499.16666666666663, 183.33333333333331], [499.16666666666663, 330.0], [3.3333333333333335, 330.0], [0.0, 253.33333333333334], [43.333333333333336, 245.0], [60.833333333333336, 273.3333333333333], [80.0, 293.3333333333333], [107.5, 307.5], [133.33333333333334, 309.16666666666663], [169.16666666666666, 295.8333333333333], [190.83333333333331, 274.1666666666667], [203.33333333333334, 252.5], [225.0, 260.0], [236.66666666666666, 254.16666666666666], [260.0, 254.16666666666666], [288.3333333333333, 253.33333333333334], [287.5, 257.5], [271.6666666666667, 265.0], [324.1666666666667, 281.6666666666667], [369.16666666666663, 274.1666666666667], [337.5, 261.6666666666667], [338.3333333333333, 257.5], [355.0, 261.6666666666667], [357.5, 257.5], [339.1666666666667, 255.0], [337.5, 240.83333333333334], [348.3333333333333, 238.33333333333334], [359.1666666666667, 248.33333333333331], [377.5, 251.66666666666666], [397.5, 248.33333333333331], [408.3333333333333, 236.66666666666666], [418.3333333333333, 220.83333333333331], [427.5, 217.5], [434.16666666666663, 215.0]]],
        "object_name": "floor",
        "positive_attributes": ["tiled", "gray", "light colored"],
        "negative_attributes": ["multicolored", "maroon", "weathered", "speckled", "carpeted"]
        '''
        results['img_prefix'] = os.path.abspath(self.data_root) + '/VG/VG_100K'
        results['img_info'] = {}
        results['img_info']['filename'] = f'{results["image_id"]}.jpg'
        x, y, w, h = results["instance_bbox"]

        # filename = os.path.abspath(self.data_root) + '/VG/VG_100K' + f'/{results["image_id"]}.jpg'
        # img = cv2.imread(filename, cv2.IMREAD_COLOR)
        # # import pdb
        # # pdb.set_trace()
        # # x1, y1, x2, y2 = int(x-w/2.), int(y-h/2), int(x+w/2), int(y+h/2)
        # x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        # img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=1)
        # os.makedirs('results/tmp', exist_ok=True)
        # cv2.imwrite('results/tmp' + f'/{idx}.jpg', img)

        results['proposals'] = np.array([x, y, x+w, y+h], dtype=np.float32).reshape(1, 4)
        results['bbox_fields'] = ['proposals']
        positive_attributes = results["positive_attributes"]
        negative_attributes = results["negative_attributes"]
        labels = np.ones(len(self.classname_maps.keys())) * 2
        for att in positive_attributes:
            att_id = self.att2id.get(att, None)
            if att_id is not None:
                labels[att_id] = 1
        for att in negative_attributes:
            att_id = self.att2id.get(att, None)
            if att_id is not None:
                labels[att_id] = 0
        results['gt_labels'] = labels.astype(np.int)

        try:
            results = self.pipeline(results)
        except Exception as e:
            print(e)
            self.error_list.add(idx)
            self.error_list.add(results['img_info']['filename'])
            print(self.error_list)
            if len(self.error_list) > 20:
                return
            if not self.test_mode:
                results = self.__getitem__(np.random.randint(0, len(self)))

        # img = results['img']
        # img_metas = results['img_metas'].data
        #
        # img = img.cpu().numpy().transpose(1, 2, 0)
        # mean, std = img_metas['img_norm_cfg']['mean'], img_metas['img_norm_cfg']['std']
        # img = (255*mmcv.imdenormalize(img, mean, std, to_bgr=True)).astype(np.uint8)
        # # import pdb
        # # pdb.set_trace()
        # box = results['proposals'].numpy()[0]
        # # print(box)
        # x1, y1, x2, y2 = box.astype(np.int)
        # img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=1)
        # os.makedirs('results/tmp', exist_ok=True)
        # cv2.imwrite('results/tmp' + f'/x{idx}.jpg', img)
        return results

    def get_labels(self):
        total_gt_list = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            img_info = self.id2images[img_id]
            instances = self.id2instances[img_id]
            data_set = img_id.split('_')[0]

            bbox_list = []
            attr_label_list = []
            for instance in instances:
                key = 'bbox' if data_set in ['coco', 'label_coco', 'label_cc3m', 'ovadattr', 'ovadcate'] else 'instance_bbox'
                x, y, w, h = instance[key]
                bbox_list.append([x, y, x + w, y + h])

                labels = np.ones(len(self.att2id)) * 2
                if data_set in ['vaw', 'ovadattr']:
                    positive_attributes = instance["positive_attributes"]
                    negative_attributes = instance["negative_attributes"]
                    for att in positive_attributes:
                        att_id = self.att2id.get(att, None)
                        if att_id is not None:
                            labels[att_id] = 1
                    for att in negative_attributes:
                        att_id = self.att2id.get(att, None)
                        if att_id is not None:
                            labels[att_id] = 0
                attr_label_list.append(labels)

            gt_bboxes = np.array(bbox_list, dtype=np.float32).reshape(-1, 4)
            gt_labels = np.stack(attr_label_list, axis=0).reshape(-1, len(self.att2id))
            total_gt_list.append(np.concatenate([gt_bboxes, gt_labels], axis=1))

        return total_gt_list

    def get_img_instance_labels(self):
        attr_label_list = []
        for img_id in self.img_ids:
            instances = self.id2instances[img_id]
            for instance in instances:
                x, y, w, h = instance["instance_bbox"]
                positive_attributes = instance.get("positive_attributes", [])
                negative_attributes = instance.get("negative_attributes", [])

                labels = np.ones(len(self.att2id.keys())) * 2

                for att in positive_attributes:
                    att_id = self.att2id.get(att, None)
                    if att_id is not None:
                        labels[att_id] = 1
                for att in negative_attributes:
                    att_id = self.att2id.get(att, None)
                    if att_id is not None:
                        labels[att_id] = 0

                attr_label_list.append(labels)
        gt_labels = np.stack(attr_label_list, axis=0)
        return gt_labels

    def get_rpn_img_instance_labels(self):
        gt_labels = []
        for img_id in self.img_ids:
            instances = self.id2instances[img_id]
            gt_labels_tmp = []
            for instance in instances:
                x, y, w, h = instance["instance_bbox"]
                positive_attributes = instance.get("positive_attributes", [])
                negative_attributes = instance.get("negative_attributes", [])

                labels = [2] * len(self.att2id.keys())

                for att in positive_attributes:
                    labels[self.att2id[att]] = 1
                for att in negative_attributes:
                    labels[self.att2id[att]] = 0

                gt_labels_tmp.append([x, y, x+w, y+h]+ labels)
            gt_labels_tmp = torch.tensor(gt_labels_tmp)
            gt_labels.append(gt_labels_tmp)
        return gt_labels

    def format_results(self, results, jsonfile_prefix=None, coco_img_ids=[], **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(coco_img_ids), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(coco_img_ids)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix, coco_img_ids)
        return result_files, tmp_dir

    def results2json(self, results, outfile_prefix, coco_img_ids):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results, coco_img_ids)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def _det2json(self, results, coco_img_ids):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(coco_img_ids)):
            img_id = coco_img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def evaluate_det_segm(self,
                          results,
                          result_files,
                          coco_gt,
                          metrics,
                          logger=None,
                          classwise=True,
                          proposal_nums=(100, 300, 1000),
                          iou_thrs=[0.5],
                          metric_items=None):
        """Instance segmentation and object detection evaluation in COCO
        protocol.

        Args:
            results (list[list | tuple | dict]): Testing results of the
                dataset.
            result_files (dict[str, str]): a dict contains json file path.
            coco_gt (COCO): COCO API object with ground truth annotation.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        eval_results = OrderedDict()
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                if isinstance(results[0], tuple):
                    raise KeyError('proposal_fast is not supported for '
                                   'instance segmentation result.')
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = mmcv.load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                    warnings.simplefilter('once')
                    warnings.warn(
                        'The key "bbox" is deleted for more accurate mask AP '
                        'of small/medium/large instances since v2.12.0. This '
                        'does not change the overall mAP calculation.',
                        UserWarning)
                coco_det = coco_gt.loadRes(predictions)
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break
            cocoEval = COCOeval(coco_gt, coco_det, iou_type)
            # cocoEval.params.catIds = self.cat_ids
            # cocoEval.params.imgIds = self.coco_img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate()

                # Save coco summarize print information to logger
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                print_log('\n' + redirect_string.getvalue(), logger=logger)

                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()

                # Save coco summarize print information to logger
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                print_log('\n' + redirect_string.getvalue(), logger=logger)

                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    seen_ids = set([self.category2id[x] for x in self.category_seen_unseen['seen']])
                    unseen_ids = set([self.category2id[x] for x in self.category_seen_unseen['unseen']])
                    results_per_category = []
                    results_per_category50 = []
                    results_per_category50_seen = []
                    results_per_category50_unseen = []
                    # import pdb
                    # pdb.set_trace()
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = coco_gt.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))
                        precision50 = precisions[0, :, idx, 0, -1]
                        precision50 = precision50[precision50 > -1]
                        ap50 = np.mean(precision50) if precision50.size else float("nan")
                        results_per_category50.append((f'{nm["name"]}', f'{float(ap50):0.3f}'))
                        if idx in seen_ids:
                            results_per_category50_seen.append(float(ap50))
                        if idx in unseen_ids:
                            results_per_category50_unseen.append(float(ap50))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                    num_columns = min(6, len(results_per_category50) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category50))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                    print(
                        "Seen {} AP50: {}".format(
                            iou_type,
                            sum(results_per_category50_seen) / len(results_per_category50_seen),
                        ))
                    print(
                        "Unseen {} AP50: {}".format(
                            iou_type,
                            sum(results_per_category50_unseen) / len(results_per_category50_unseen),
                        ))

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                    )
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')
                print(eval_results)
        return eval_results
    def eval_coco_det(self, gt_boxes, coco_img_ids, results, jsonfile_prefix=None):
        self.cat_ids = {v: v for k, v in self.category2id.items()}

        coco_gt = COCO()
        coco_gt.dataset['images'] = [copy.deepcopy(self.id2images[x]) for x in coco_img_ids]
        for idx in range(len(coco_gt.dataset['images'])):
            coco_gt.dataset['images'][idx]['id'] = 'coco_' + str(coco_gt.dataset['images'][idx]['id'])
        coco_gt_cates = [{'id': v, "name": k, 'supercategory': 'none'} for k, v in self.category2id.items()]
        coco_gt.dataset['categories'] = coco_gt_cates

        ann_id = 1
        refined_boxes = []
        for id, anns in enumerate(gt_boxes):
            image_id = anns['image_id']
            for box, label in zip(anns['bboxes'], anns['labels']):
                data = dict()
                data['image_id'] = image_id
                x1, y1, x2, y2 = [box[0], box[1], box[2], box[3]]
                data['bbox'] = [x1, y1, x2-x1, y2-y1]
                data['category_id'] = self.cat_ids[label]
                data['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                data['area'] = data['bbox'][2] * data['bbox'][3]
                data['id'] = ann_id
                data['iscrowd'] = 0
                ann_id += 1
                refined_boxes.append(data)
        coco_gt.dataset['annotations'] = refined_boxes
        coco_gt.createIndex()

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix, coco_img_ids)
        eval_results = self.evaluate_det_segm(results, result_files, coco_gt,
                                              metrics=['bbox'])
        return eval_results
    def evaluate(self,
                 results,
                 nms_cfg=None,
                 logger=None,
                 metric='mAP',
                 per_class_out_file=None,
                 is_logit=True,
                 ):
        pass

