import copy
import glob
import json
import logging
import os
import os.path as osp
import pickle
import random
import tempfile
import warnings
from collections import OrderedDict, defaultdict
from pprint import pprint

import cv2
import imagesize
import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer
from mmcv.runner import get_dist_info
from sklearn import metrics

from tools_my.cache_data_tools.redis_utils import RedisHelper
from ..datasets.builder import DATASETS
from torch.utils.data import Dataset
from ..datasets.pipelines import Compose
from .evaluate_tools import cal_metrics


@DATASETS.register_module()
class BoostCLIPWithCapCropDataset(Dataset):
    CLASSES = None

    def __init__(self,
                 data_root,
                 cap_pipeline,
                 vawcoco_pipline=None,
                 dataset_split='train',
                 attribute_index_file=None,
                 test_mode=False,
                 open_category=True,
                 dataset_names='vaw',
                 load_label=None,
                 save_label=False,
                 select_novel=False,
                 file_client_args=dict(backend='disk')
                 ):
        assert dataset_split in ['train']
        self.dataset_split = dataset_split
        self.test_mode = test_mode
        if isinstance(cap_pipeline, list):
            # train_cap_wholeimg_pipeline, train_cap_biggestproposal_pipeline,
            # train_cap_imgcrops_pipeline, train_cap_collectall_pipeline
            self.cap_pipeline = [Compose(x) for x in cap_pipeline]
        else:
            self.cap_pipeline = Compose(cap_pipeline)
        if vawcoco_pipline is not None:
            self.vawcoco_pipline = Compose(vawcoco_pipline)
        self.data_root = data_root
        self.dataset_names = dataset_names
        self.select_novel = select_novel

        self.attribute_index_file = attribute_index_file
        self.att2id = {}
        self.att_seen_unseen = {}
        if 'att_file' in attribute_index_file.keys():
            file = attribute_index_file['att_file']
            att2id = json.load(open(file, 'r'))
            att_group = attribute_index_file['att_group']
            if att_group in ['common1', 'common2', 'common', 'rare']:
                self.att2id = att2id[att_group]
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
            self.id2instances.pop('vaw_713545', None)
            self.id2instances.pop('vaw_2369080', None)

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

        if 'cococap' in self.dataset_names:
            id2images_coco, id2instances_coco = self.read_data_coco_cap(dataset_split)
            self.id2images.update(id2images_coco)
            self.id2instances.update(id2instances_coco)
            self.redis_helper = RedisHelper()

        self.instances = []
        for k, v in self.id2instances.items():
            for item in v:
                item['img_id'] = k
                self.instances.append(item)
        if 'cc3m' in self.dataset_names:
            keys = json.load(open('../attributes/CC/train_CC_keys.json', 'r'))
            for k in keys:
                self.instances.append({'img_id': f'cc3m_{k}'})

        rank, world_size = get_dist_info()
        if not test_mode:
            self.instances = self.filter_instance(self.instances)
            flag_dataset = [x['img_id'].split('_')[0] for x in self.instances]
            dataset_types = {'coco': 0, 'vaw': 1, 'ovadgen': 1, 'cococap': 2, 'cc3m': 2}
            flag_dataset = [dataset_types[x] for x in flag_dataset]
            # self.flag_dataset = np.array(flag_dataset, dtype=np.int)

        self.flag = np.zeros(len(self), dtype=int)

        if rank == 0:
            print(np.bincount(flag_dataset))
            print('data len: ', len(self))
            print('num_att: ', len(self.att2id))
            print('num_category: ', len(self.category2id))
        self.error_list = set()

        self.save_label = save_label
        if load_label:
            self.pred_labels = np.load(load_label)
            assert len(self) == len(self.pred_labels)

    def filter_instance(self, instances, filter_func='seen'):
        return_instances = []
        for instance in instances:
            img_id = instance['img_id']
            data_set = img_id.split('_')[0]
            if data_set == 'coco':
                category = instance['name']
                if filter_func == 'seen':
                    if category in self.category_seen_unseen['seen']:
                        return_instances.append(instance)
                else:
                    category_id = self.category2id.get(category, None)  # 未标注的该类别的应该去除
                    if category_id is not None:
                        return_instances.append(instance)
            elif data_set == 'vaw':
                return_instances.append(instance)
            elif data_set == 'ovadgen':
                return_instances.append(instance)
            elif data_set == 'cococap':
                return_instances.append(instance)
            elif data_set == 'cc3m':
                return_instances.append(instance)
        return return_instances

    def read_data_ovadgen(self, pattern):
        instances = glob.glob(self.data_root + '/ovadgen/*.jpg')
        instances = [x for x in instances if os.path.getsize(x) > 25 * 1024]

        id2images = {}
        id2instances = {}
        for idx, data in enumerate(instances):
            img_id = 'ovadgen_' + str(idx)
            id2images[img_id] = {}
            id2images[img_id]['file_name'] = os.path.basename(data)
        id2att = {v: k for k, v in self.att2id.items()}
        for idx, data in enumerate(instances):
            img_id = 'ovadgen_' + str(idx)
            instance = {}
            img_name = os.path.basename(data)
            id_att = int(img_name.split('_')[-2])
            att_name = '_'.join(img_name.split('_')[:-2])
            target_att = id2att[id_att]

            att_type, att_names = target_att.split(':')
            target_atts = [x + ' ' + att_type for x in att_names.split('/')]
            target_atts = [x.replace(' ', '_') for x in target_atts]
            assert att_name in target_atts
            instance['positive_attributes'] = [target_att]
            instance['negative_attributes'] = []
            id2instances[img_id] = id2instances.get(img_id, []) + [instance]

        return id2images, id2instances

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

    def read_data_coco_cap(self, pattern):
        assert pattern == 'train'
        json_file = 'instances_train2017'
        # json_file = 'lvis_v1_train' if pattern == 'train' else 'instances_val2017'
        json_data = json.load(open(self.data_root + f'/COCO/annotations/{json_file}.json', 'r'))

        id2images = {}
        id2instances = {}
        for data in json_data['images']:
            img_id = 'cococap_' + str(data['id'])
            data['file_name'] = f'{data["id"]:012d}.jpg'
            id2images[img_id] = data
        cap_data = json.load(open(self.data_root + f'/COCO/annotations/train_2017_caption_tagging_with_proposals.json', 'r'))
        for img_id, data in cap_data.items():
            if self.select_novel:
                keep_flag = False
                for cate in self.category_seen_unseen['unseen']:
                    if cate in data['category']:
                        keep_flag = True
                        break
                for att in self.att_seen_unseen['unseen']:
                    if keep_flag:
                        break
                    if att in data['attribute']:
                        keep_flag = True
            else:
                keep_flag = True
            if keep_flag:
                img_id = 'cococap_' + str(img_id)
                id2instances[img_id] = id2instances.get(img_id, []) + [data]
        return id2images, id2instances

    def read_data_vaw(self, pattern):
        json_files = ['train_part1', 'train_part2'] if pattern == 'train' else [f'{pattern}']
        json_data = [json.load(open(self.data_root + '/VAW/' + f'{x}.json', 'r')) for x in json_files]
        instances = []
        [instances.extend(x) for x in json_data]
        # instances = instances[:1024]
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

    def read_data_ovad(self, pattern):
        json_data = json.load(open('../attributes/OVAD/ovad1200_licence.json', 'r'))
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

    def split_instance_by_category(self, pattern='train'):
        categories = json.load(open(self.data_root + '/VAW/' + 'category_instances_split.json'))[f'{pattern}_category']
        categories = [x[0] for x in categories]
        instances = []
        for instance in self.instances:
            if instance['object_name'] in categories:
                instances.append(instance)
        return instances

    def __len__(self):
        return len(self.instances)

    def get_generated_sample(self, idx):
        instance = self.instances[idx]
        results = {}
        results['img_prefix'] = ''
        results['img_info'] = {}
        results['img_info']['filename'] = instance
        labels = np.ones(len(self.att2id.keys())) * 2
        att = ' '.join(os.path.basename(instance).split('_')[:-2])
        att_id = self.att2id.get(att, None)
        labels[att_id] = 1
        if hasattr(self, 'pred_labels'):
            thresh_low = 0.1
            thresh_high = 0.5
            thresh_topk = 3
            pred_label = torch.from_numpy(self.pred_labels[idx])
            idx_tmp = torch.nonzero(pred_label < thresh_low)[:, 0]
            labels[idx_tmp] = 0
            # values, idx_tmp = torch.topk(-pred_label, k=thresh_topk)
            # labels[idx_tmp] = 0
            # idx_tmp = torch.nonzero(pred_label > thresh_high)[:, 0]
            # labels[idx_tmp] = 1
            # values, idx_tmp = torch.topk(pred_label, k=thresh_topk)
            # labels[idx_tmp] = 1

        results['gt_labels'] = labels.astype(np.int)
        results = self.pipeline(results)
        return results

    def __getitem__(self, idx):
        if idx in self.error_list and not self.test_mode:
            idx = np.random.randint(0, len(self))
        instance = copy.deepcopy(self.instances[idx])
        img_id = instance['img_id']
        if 'cc3m' in self.dataset_names:
            img_info = dict()
            img_info['file_name'] = f'{img_id.split("_")[-1]}.jpg'
        else:
            img_info = self.id2images[img_id]

        data_set, img_id = img_id.split('_')
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
        elif data_set == 'cococap':
            data_set_type = 2
            prefix_path = f'/COCO/train2017'
            if not self.redis_helper.redis:
                self.redis_helper.init_redis()
            instance.update(self.redis_helper.get_values('cky_'+img_id))
        elif data_set == 'cc3m':
            data_set_type = 2
            sub_folder = img_id[:5]
            prefix_path = f'/CC3M/CC3MImages/{sub_folder}/'
        else:
            raise NameError
        results = {}
        results['data_set_type'] = data_set_type
        results['img_prefix'] = os.path.abspath(self.data_root) + prefix_path
        results['img_info'] = {}
        results['img_info']['filename'] = img_info['file_name']

        if data_set == 'cc3m':
            instance_tmp = json.load(open(
                os.path.abspath(self.data_root) + f'/CC3M/proposals_labels/{img_id}.json', 'r'))
            instance.update(instance_tmp)
            instance_tmp = json.load(open(
                os.path.abspath(self.data_root) + f'/CC3M/tagging_labels/{img_id}.json', 'r'))
            instance.update(instance_tmp)
            instance['caption'] = [instance['caption']]

        # get whole img
        results_tmp = copy.deepcopy(self.cap_pipeline[0](results, 0))

        # 全图，不需要
        # results_wholeimg = copy.deepcopy(results)
        # results_wholeimg = self.cap_pipeline[0](results_wholeimg, (1, ':'))
        # results['img'] = results_wholeimg['img']

        # get biggest proposal
        results_biggestproposal = copy.deepcopy(results_tmp)
        # 注意xyxy还是xywh
        x, y, w, h = instance['biggest_proposal'][:4]
        results_biggestproposal['crop_box'] = np.array([x, y, x + w, y + h])
        results_biggestproposal = self.cap_pipeline[1](results_biggestproposal)
        results['img'] = results_biggestproposal['img']

        # get label
        labels = np.ones(len(self.att2id) + len(self.category2id)) * 2
        labels[len(self.att2id):] = 0
        positive_attributes = instance["attribute"]
        for att in positive_attributes:
            att_id = self.att2id.get(att, None)
            if att_id is not None:
                labels[att_id] = 1
        categories = instance["category"]
        for category in categories:
            category_id = self.category2id.get(category, None)
            if category_id is not None:
                labels[category_id + len(self.att2id)] = 1
        results['gt_labels'] = labels.astype(np.int)

        # get random max_crops img crops and crossponding teacher logits
        # COCOCap att_thres = 0.65
        #         cate_thres = 0.7
        # CC3M  att_thres = 0.65
        #       cate_thres = 0.65
        max_crops = 5
        att_thres = 0.7
        cate_thres = 0.7
        img_crops = []
        crops_logits = []
        crops_labels = []

        proposals = np.array(instance['proposals'])
        areas = proposals[:, 2] * proposals[:, 3]
        proposals = proposals[areas > 16]
        confidences = proposals[:, 4]
        proposals = proposals[confidences > 0.1]
        proposals_inds = list(range(0, len(proposals)))
        random.shuffle(proposals_inds)
        for proposal_id in proposals_inds:
            if len(img_crops) >= max_crops:
                break
            teacher_logits = torch.from_numpy(proposals[proposal_id][6:]).float()

            cate_in_img = instance['category']
            att_in_img = instance['attribute']

            assert len(teacher_logits) == len(self.att2id) + len(self.category2id)

            teacher_att = teacher_logits[:len(self.att2id)].sigmoid()
            teacher_cate = teacher_logits[len(self.att2id):].softmax(dim=0)

            pesu_label_att = torch.ones_like(teacher_att) * 2
            pesu_label_att[teacher_att > att_thres] = 1
            for att in att_in_img:
                att_id = self.att2id.get(att, None)
                if att_id is not None:
                    if teacher_att[att_id] > 0.52:
                        pesu_label_att[att_id] = 1

            pesu_label_cate = teacher_cate > cate_thres
            for cate in cate_in_img:
                cate_id = self.category2id.get(cate, None)
                if cate_id is not None:
                    if teacher_cate[cate_id] > 0.52:
                        pesu_label_att[cate_id] = 1
            if torch.any(pesu_label_cate > 0):
                results_img_crops = copy.deepcopy(results_tmp)
                x, y, w, h = proposals[proposal_id][:4]  # xywh,c,c,621a
                results_img_crops['crop_box'] = np.array([x, y, x + w, y + h])
                cap_imgcrops = self.cap_pipeline[2](results_img_crops)
                img_crops.append(cap_imgcrops['img'])
                crops_logits.append(teacher_logits)
                crops_labels.append(torch.cat((pesu_label_att, pesu_label_cate)))
        if len(img_crops) == 0:
            # print(img_id, ': error')
            self.error_list.add(idx)
            return self.__getitem__(np.random.randint(0, len(self)))
        img_crops = torch.stack(img_crops, dim=0)
        results['img_crops'] = img_crops
        results['crops_logits'] = torch.stack(crops_logits, dim=0)
        results['crops_labels'] = torch.stack(crops_labels, dim=0)

        # get random select caption
        random_id = random.randint(0, len(instance['caption']) - 1)
        results['caption'] = DataContainer(instance['caption'][random_id], stack=True, cpu_only=True)

        # get dataset type
        results['data_set_type'] = data_set_type

        # get all phases
        phases = instance['phase']
        # COCO Cap = 5
        # CC 3M = 3
        max_phase = 5
        if len(phases) > max_phase:
            random_id = [random.randint(0, len(phases) - 1) for _ in range(max_phase)]
            phases = [phases[x] for x in random_id]

        # # get random select caption
        # if len(phases) == 0:
        #     samp_phase = 'a image.'
        # else:
        #     random_id = random.randint(0, len(phases) - 1)
        #     samp_phase = phases[random_id]
        # results['samp_phase'] = DataContainer(samp_phase, stack=True, cpu_only=True)

        results['phases'] = DataContainer(phases, stack=False, cpu_only=True)
        results['img_crops'] = DataContainer(results['img_crops'], stack=False)
        results['crops_logits'] = DataContainer(results['crops_logits'], stack=False)
        results['crops_labels'] = DataContainer(results['crops_labels'], stack=False)
        results = self.cap_pipeline[3](results)
        return results



    def get_labels(self):
        np_gt_labels = []
        for instance in self.instances:
            labels = np.ones(len(self.att2id) + len(self.category2id)) * 2
            labels[len(self.att2id):] = 0
            img_id = instance['img_id']
            img_info = self.id2images[img_id]
            data_set = img_id.split('_')[0]
            # import pdb
            # pdb.set_trace()
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
            if data_set in ['coco', 'ovadcate']:
                category = instance['name']
                category_id = self.category2id.get(category, None)
                if category_id is not None:
                    labels[category_id + len(self.att2id)] = 1
            np_gt_labels.append(labels.astype(np.int))
        return np.stack(np_gt_labels, axis=0)

    def get_data_set_type(self):
        data_set_types = []
        for instance in self.instances:
            img_id = instance['img_id']
            img_info = self.id2images[img_id]
            data_set = img_id.split('_')[0]
            if data_set == 'coco':
                data_set_type = 0
            elif data_set == 'vaw':
                data_set_type = 1
            elif data_set == 'ovadcate':
                data_set_type = 0
            elif data_set == 'ovadattr':
                data_set_type = 1
            else:
                raise NameError
            data_set_types.append(data_set_type)
        return np.array(data_set_types)

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 per_class_out_file=None,
                 is_logit=True
                 ):
        pass


