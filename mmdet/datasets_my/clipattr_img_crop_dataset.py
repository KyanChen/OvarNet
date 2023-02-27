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
from mmcv.parallel import DataContainer

import cv2
import imagesize
import mmcv
import numpy as np
import torch
from mmcv.runner import get_dist_info
from sklearn import metrics
from tqdm import tqdm

from tools_my.cache_data_tools.redis_utils import RedisHelper
from ..core import bbox_overlaps
from ..datasets.builder import DATASETS
from torch.utils.data import Dataset
from ..datasets.pipelines import Compose
from .evaluate_tools import cal_metrics


@DATASETS.register_module()
class CLIPAttrImgCropDataset(Dataset):

    CLASSES = None

    def __init__(self,
                 data_root,
                 pipeline,
                 dataset_split='train',
                 attribute_index_file=None,
                 test_mode=False,
                 open_category=True,
                 dataset_names='vaw',
                 load_label=None,
                 save_label=False,
                 ):
        super(CLIPAttrImgCropDataset, self).__init__()

        assert dataset_split in ['train', 'val', 'test']
        self.dataset_split = dataset_split
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.data_root = data_root
        self.dataset_names = dataset_names

        self.attribute_index_file = attribute_index_file
        self.att2id = {}
        self.att_seen_unseen = {}
        if 'att_file' in attribute_index_file.keys():
            file = attribute_index_file['att_file']
            att2id = json.load(open(file, 'r'))
            att_group = attribute_index_file['att_group']
            if att_group in ['common1', 'common2', 'common', 'rare', 'base']:
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

        if open_category:
            print('open_category: ', open_category)
            self.instances, self.img_instances_pair = self.read_data(["train_part1.json", "train_part2.json", 'val.json', 'test.json'])
            self.instances = self.split_instance_by_category(pattern=pattern)
        else:
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

            if 'label_coco' in self.dataset_names:
                dataset_split = 'train'
                id2images_coco, id2instances_coco = self.read_data_coco(dataset_split)
                self.id2images.update(id2images_coco)
                proposal_json = json.load(open(self.data_root + f'/COCO/annotations/train_2017_caption_tagging_with_proposals.json', 'r'))
                id2instances = {}
                for image_id, v in proposal_json.items():
                    img_id = 'coco_' + str(image_id)
                    for box in v['proposals']:
                        data = {'bbox': box[:4]}
                        id2instances[img_id] = id2instances.get(img_id, []) + [data]
                self.id2instances.update(id2instances)

            if 'label_cc3m' in self.dataset_names:
                keys = json.load(open('../attributes/CC/train_CC_keys.json', 'r'))

                id2images_cc3m = {}
                for k in keys:
                    # data_json = json.load(
                    #     open(f'/data/kyanchen/prompt/data/CC3M/tagging_labels/{k}.json'))
                    # img_info = dict()
                    # img_info['file_name'] = f'{k}.jpg'
                    # img_info['width'] = data_json['width']
                    # img_info['height'] = data_json['height']
                    # id2images_cc3m[f'cc3m_{k}'] = img_info
                    id2images_cc3m[f'labelcc3m_{k}'] = {}
                self.id2images.update(id2images_cc3m)

                # id2instances = {}
                # for image_id, v in id2images_cc3m.items():
                #     img_key = image_id.split('_')[-1]
                #     proposal_json = json.load(
                #         open(f'/data/kyanchen/prompt/data/CC3M/proposals/{img_key}.json', 'r')
                #     )
                #     for box in proposal_json['proposals']:
                #         data = {'bbox': box[:4]}
                #         id2instances[image_id] = id2instances.get(image_id, []) + [data]
                self.id2instances.update(id2images_cc3m)

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

            self.instances = []
            for k, v in self.id2instances.items():
                for item in v:
                    item['img_id'] = k
                    self.instances.append(item)
            # self.instances = self.instances[:512]

        rank, world_size = get_dist_info()
        if not test_mode:
            self.instances = self.filter_instance(self.instances)
            flag_dataset = [x['img_id'].split('_')[0] for x in self.instances]
            dataset_types = {'coco': 0, 'vaw': 1, 'ovadgen': 1}
            flag_dataset = [dataset_types[x] for x in flag_dataset]
            self.flag_dataset = np.array(flag_dataset, dtype=np.int)

        self.flag = np.zeros(len(self), dtype=int)

        if rank == 0:
            print('data len: ', len(self))
            print('num_att: ', len(self.att2id))
            print('num_category: ', len(self.category2id))
        self.error_list = set()

        self.save_label = save_label
        if load_label:
            self.pred_labels = np.load(load_label)
            assert len(self) == len(self.pred_labels)

    def filter_instance(self, instances):
        return_instances = []
        for instance in instances:
            img_id = instance['img_id']
            data_set = img_id.split('_')[0]
            if data_set == 'coco':
                category = instance['name']
                category_id = self.category2id.get(category, None)  # 未标注的该类别的应该去除
                if category_id is not None:
                    return_instances.append(instance)
            elif data_set == 'vaw':
                return_instances.append(instance)
            elif data_set == 'ovadgen':
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

    def split_instance_by_category(self, pattern='train'):
        categories = json.load(open(self.data_root + '/VAW/' + 'category_instances_split.json'))[f'{pattern}_category']
        categories = [x[0] for x in categories]
        instances = []
        for instance in self.instances:
            if instance['object_name'] in categories:
                instances.append(instance)
        return instances

    def __len__(self):
        if 'label_cc3m' in self.dataset_names:
            return len(self.id2instances)
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
        if 'label_cc3m' in self.dataset_names:
            img_id = list(self.id2images.keys())[idx]
            img_info = dict()
            img_info['file_name'] = f'{img_id.split("_")[-1]}.jpg'
        else:
            instance = self.instances[idx]
            img_id = instance['img_id']
            img_info = self.id2images[img_id]

        data_set, img_key = img_id.split('_')
        if data_set in ['coco', 'labelcoco']:
            data_set_type = 0
            if self.dataset_split == 'test':
                dataset_split = 'val'
            else:
                dataset_split = self.dataset_split
            prefix_path = f'/COCO/{dataset_split}2017'
        elif data_set == 'labelcc3m':
            data_set_type = 0
            sub_folder = img_key[:5]
            prefix_path = f'/CC3M/CC3MImages/{sub_folder}/'
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
        if 'gen' in data_set:
            pass
        elif data_set == 'labelcc3m':
            try:
                proposal_json = json.load(
                    open(f'/data/kyanchen/prompt/data/CC3M/proposals/{img_key}.json', 'r')
                )
            except:
                print(img_key)
            instance_proposal = np.array(proposal_json['proposals'])[:, :4]
            instance_proposal[:, 2] = instance_proposal[:, 0] + instance_proposal[:, 2]
            instance_proposal[:, 3] = instance_proposal[:, 1] + instance_proposal[:, 3]
        else:
            key = 'bbox' if data_set in ['coco', 'label_coco', 'label_cc3m', 'ovadattr', 'ovadcate'] else 'instance_bbox'
            x, y, w, h = instance[key]
            results['crop_box'] = np.array([x, y, x + w, y + h])
        if self.test_mode:
            if data_set == 'labelcc3m':
                results = self.pipeline(results, 0)
                crop_imgs = []
                for idx, proposal in enumerate(instance_proposal):
                    results_tmp = copy.deepcopy(results)
                    results_tmp['crop_box'] = proposal
                    try:
                        results_tmp = self.pipeline(results_tmp, [1, ':'])
                    except:
                        print('xxxx:', img_key)
                        results_tmp = copy.deepcopy(results)
                        results_tmp['crop_box'] = instance_proposal[idx-1]
                        results_tmp = self.pipeline(results_tmp, [1, ':'])
                    crop_imgs.append(results_tmp['img'])
                crop_imgs = torch.stack([x[0].data for x in crop_imgs], dim=0)
                results['img'] = crop_imgs
                extra_pip = dict(type='MultiScaleFlipAug',
                                 img_scale=(224, 224),
                                 flip=False,
                                 transforms=[
                                     dict(type='Collect',
                                          meta_keys=('filename',
                                                     'ori_filename',
                                                     'ori_shape',
                                                     'img_shape'), keys=['img', 'img_key'])
                                ])

                extra_pip = Compose([extra_pip])
                results['img_key'] = DataContainer(img_key, stack=True, cpu_only=True)
                results = extra_pip(results)
            else:
                results = self.pipeline(results)
            # try:
            #
            # except Exception as e:
            #     print(f'idx: {idx}')
            #     print(f'img_id: {img_id}')
        else:
            try:
                labels = np.ones(len(self.att2id)+len(self.category2id)) * 2
                labels[len(self.att2id):] = 0
                if data_set == 'vaw' or data_set == 'ovadgen':
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
                if data_set == 'coco':
                    category = instance['name']
                    category_id = self.category2id.get(category, None)
                    if category_id is not None:
                        labels[category_id+len(self.att2id)] = 1
                results['gt_labels'] = labels.astype(np.int)
                if 'gen' in data_set:
                    results = self.pipeline(results, 0)
                    results = self.pipeline(results, (2, ':'))
                else:
                    results = self.pipeline(results)
            except Exception as e:
                self.error_list.add(idx)
                self.error_list.add(img_id)
                print(self.error_list)
                results = self.__getitem__(np.random.randint(0, len(self)))

        # img = results['img']
        # img_metas = results['img_metas'].data
        #
        # img = img.cpu().numpy().transpose(1, 2, 0)
        # mean, std = img_metas['img_norm_cfg']['mean'], img_metas['img_norm_cfg']['std']
        # img = (255*mmcv.imdenormalize(img, mean, std, to_bgr=True)).astype(np.uint8)
        #
        # os.makedirs('results/tmp', exist_ok=True)
        # cv2.imwrite('results/tmp' + f'/x{idx}.jpg', img)
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
        result_metrics = OrderedDict()

        results = np.array(results)
        if 'label_coco' in self.dataset_names:
            ori_data = json.load(
                open(self.data_root + f'/COCO/annotations/train_2017_caption_tagging_with_proposals.json', 'r'))
            flag_id_start = 0
            pred_atts = results
            redis_helper = RedisHelper()
            if not redis_helper.redis:
                redis_helper.init_redis()
            prefix = 'cky_'
            mmcv.dump(results, f'train2017_proposals_predatts_vit16.pkl', protocol=4)
            for img_id, data in tqdm(ori_data.items()):
                proposals = np.array(data['proposals'])
                flag_id_end = flag_id_start + len(proposals)
                proposals_atts = pred_atts[flag_id_start: flag_id_end]
                proposals = np.concatenate((proposals, proposals_atts), axis=-1)  # 6 [xywh,conf,class]+ 606
                data['proposals'] = proposals.tolist()
                data['img_id'] = img_id
                img_id = prefix + str(img_id)
                redis_helper.redis.set(img_id, json.dumps(data))
                flag_id_start = flag_id_end

        preds = torch.from_numpy(results)
        gt_labels = self.get_labels()
        gt_labels = torch.from_numpy(gt_labels)

        data_set_type = self.get_data_set_type()
        data_set_type = torch.from_numpy(data_set_type)
        # import pdb
        # pdb.set_trace()
        cate_mask = data_set_type == 0
        att_mask = data_set_type == 1
        pred_att_logits = preds[att_mask][:, :len(self.att2id)]
        pred_cate_logits = preds[cate_mask][:, len(self.att2id):]
        gt_att = gt_labels[att_mask][:, :len(self.att2id)]
        gt_cate = gt_labels[cate_mask][:, len(self.att2id):]
        # import pdb
        # pdb.set_trace()
        if len(pred_cate_logits):
            dataset_name = self.attribute_index_file['category_file'].split('/')[-2]
            top_k = 1 if dataset_name == 'COCO' else -1

            pred_cate_logits = pred_cate_logits.detach().sigmoid().cpu()
            # pred_cate_logits = pred_cate_logits.float().softmax(dim=-1).cpu()
            #
            #         if self.mult_proposal_score:
            #             proposal_scores = [p.get('objectness_logits') for p in proposals]
            #             scores = [(s * ps[:, None]) ** 0.5 \
            #                 for s, ps in zip(scores, proposal_scores)]
            pred_cate_logits = pred_cate_logits * (pred_cate_logits == pred_cate_logits.max(dim=-1)[0][:, None])
            gt_cate = gt_cate.detach().cpu()

            # values, indices = torch.max(pred_cate_logits, dim=-1)
            # row_indices = torch.arange(len(values))[values > 0.5]
            # col_indices = indices[values > 0.5]
            # pred_cate_logits[row_indices, col_indices] = 1
            # pred_cate_logits[pred_cate_logits < 1] = 0

            pred_cate_logits = pred_cate_logits.numpy()
            gt_cate = gt_cate.numpy()

            output = cal_metrics(
                self.category2id,
                dataset_name,
                prefix_path=f'../attributes/{dataset_name}',
                pred=pred_cate_logits,
                gt_label=gt_cate,
                top_k=top_k,
                save_result=True,
                att_seen_unseen=self.category_seen_unseen
            )
            # import pdb
            # pdb.set_trace()
            # print(output)
            result_metrics['cate_ap_all'] = output['PC_ap/all']

            # pred_pos_mask = pred_prob > 0.5
            # pred_neg_mask = pred_prob <= 0.5
            #
            # tp = torch.sum(gt_cate[pred_pos_mask][torch.arange(len(gt_cate[pred_pos_mask])), pred_label[pred_pos_mask]] == 1)
            # tn = torch.sum(torch.sum(gt_cate[pred_neg_mask], dim=-1) == 0)
            # fp = torch.sum(gt_cate[pred_pos_mask][torch.arange(len(gt_cate[pred_pos_mask])), pred_label[pred_pos_mask]] == 0)
            # fn = torch.sum(torch.sum(gt_cate[pred_neg_mask], dim=-1) == 1)
            #
            # result_metrics['cate_precision'] = tp / torch.sum(pred_pos_mask)
            # result_metrics['cate_recall'] = tp / (tp + fn)
            # result_metrics['cate_acc'] = (tp + tn) / (tp + tn + fp + fn)
            # result_metrics['cate_f1'] = 2 * result_metrics['cate_precision'] * result_metrics['cate_recall'] / (result_metrics['cate_precision'] + result_metrics['cate_recall'])
            # result_metrics['cate_num_tp'] = tp
            # result_metrics['cate_num_tn'] = tn
            # result_metrics['cate_num_fp'] = fp
            # result_metrics['cate_num_fn'] = fn
            # print()
            # for k, v in result_metrics.items():
            #     value = f'{v.item():.4f}' if 'num' not in k else f'{v.item()}'
            #     print(k, '\t', value)
            #     result_metrics[k] = v.item()

        if self.save_label:
            np.save(self.save_label, preds.data.cpu().float().sigmoid().numpy())

        assert pred_att_logits.shape[-1] == gt_att.shape[-1]

        if not len(self.att2id):
            return result_metrics

        dataset_name = self.attribute_index_file['att_file'].split('/')[-2]
        top_k = 15 if dataset_name == 'VAW' else 8

        pred_att_logits = pred_att_logits.data.cpu().float().sigmoid().numpy()  # Nx620
        gt_att = gt_att.data.cpu().float().numpy()  # Nx620

        prs = []
        for i_att in range(pred_att_logits.shape[1]):
            y = gt_att[:, i_att]
            pred = pred_att_logits[:, i_att]
            gt_y = y[~(y == 2)]
            pred = pred[~(y == 2)]
            pr = metrics.average_precision_score(gt_y, pred)
            prs.append(pr)
        print('map: ', np.mean(prs))

        output = cal_metrics(
            self.att2id,
            dataset_name,
            prefix_path=f'../attributes/{dataset_name}',
            pred=pred_att_logits,
            gt_label=gt_att,
            top_k=top_k,
            save_result=True,
            att_seen_unseen=self.att_seen_unseen
        )
        result_metrics['att_ap_all'] = output['PC_ap/all']
        return result_metrics


