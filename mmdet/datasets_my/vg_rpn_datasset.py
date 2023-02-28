import json
import logging
import os
import os.path as osp
import pickle
import random
import tempfile
import warnings
from collections import OrderedDict, defaultdict
import contextlib
import io
import itertools
import logging
import tempfile
import warnings
from collections import OrderedDict

import cv2
import imagesize
from mmcv.parallel import DataContainer
from terminaltables import AsciiTable

import mmcv
import numpy as np
import torch
from mmcv import print_log

from mmdet.core import eval_recalls
from mmdet.utils.api_wrappers import COCO, COCOeval
from ..datasets.builder import DATASETS
from torch.utils.data import Dataset
from ..datasets.pipelines import Compose


# 得到VG的instance，并训练RPN，同时使用训练好的RPN计算VG的召回指标
@DATASETS.register_module()
class VGRPNDataset(Dataset):

    CLASSES = None

    def __init__(self,
                 data_root,
                 pipeline,
                 pattern,
                 test_mode=False,
                 file_client_args=dict(backend='disk')
                 ):
        super(VGRPNDataset, self).__init__()
        assert pattern in ['train', 'val', 'test']
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.data_root = data_root
        self.img_instances_pair = self.read_data(pattern)

        # self.error_list = set()
        self.img_ids = list(self.img_instances_pair.keys())
        if pattern == "train":
            self._set_group_flag()
        print('data len: ', len(self.img_instances_pair))

    def xyxy2xywh(self, bbox):
        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def format_gt(self):
        tmp_dir = tempfile.TemporaryDirectory()
        jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = [instance['instance_bbox'] for instance in self.img_instances_pair[img_id]]
            bboxes = np.array(bboxes).reshape(-1, 4)
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = bboxes[i]
                data['score'] = 1
                data['category_id'] = 1
                json_results.append(data)

        result_files = f'{jsonfile_prefix}.proposal.json'

        mmcv.dump(json_results, result_files)
        return json_results, result_files

    def _set_group_flag(self):
        self.flag = []
        flags_path = os.path.abspath(self.data_root) + '/VG/img_read_flags.json'
        if os.path.exists(flags_path):
            self.flag = json.load(open(flags_path, 'r'))
        if len(self.flag) != len(self.img_ids):
            self.flag = [0]*len(self)
            for i, img_id in enumerate(self.img_ids):
                img_path = os.path.abspath(self.data_root) + '/VG/VG_100K' + f'/{img_id}.jpg'
                w, h = imagesize.get(img_path)
                if w / h > 1:
                    self.flag[i] = 1
            json.dump(self.flag, open(flags_path, 'w'))
        self.flag = np.array(self.flag, dtype=np.uint8)

    def read_data(self, pattern):
        json_data = json.load(open(self.data_root + '/VG/objects.json', 'r'))
        img_ids_file = 'VG_train' if pattern == 'train' else pattern+'_image_ids'
        img_ids = json.load(open(self.data_root + f'/VAW/{img_ids_file}.json', 'r'))
        img_instances_pair_all = {}
        for data in json_data:
            img_instances_pair_all[data['image_id']] = data['objects']
        img_instances_pair = {}
        for img_id in img_ids:
            instances = []
            for instance in img_instances_pair_all[img_id]:
                x, y, w, h = instance["x"], instance["y"], instance["w"], instance["h"]
                if w * h > 0:
                    instances.append(instance)
            if len(instances):
                img_instances_pair[img_id] = instances
        return img_instances_pair

    def __len__(self):
        return len(self.img_instances_pair)

    def get_data(self, idx):
        img_id = self.img_ids[idx]
        instances = self.img_instances_pair[img_id]
        results = {}
        results['img_prefix'] = os.path.abspath(self.data_root) + '/VG/VG_100K'
        results['img_info'] = {}
        results['img_info']['filename'] = f'{img_id}.jpg'

        bbox_list = []
        for instance in instances:
            x, y, w, h = instance["x"], instance["y"], instance["w"], instance["h"]
            x = 0 if x < 0 else x
            y = 0 if y < 0 else y
            bbox_list.append([x, y, x + w, y + h])

        gt_bboxes = np.array(bbox_list, dtype=np.float32).reshape((-1, 4))
        assert len(gt_bboxes) > 0, f'w:{w}, h:{h} gt_bboxes is None'
        results['gt_bboxes'] = gt_bboxes
        results['bbox_fields'] = ['gt_bboxes']
        try:
            results = self.pipeline(results)
            # results['gt_bboxes'] = DataContainer(results['proposals'], stack=False)
            # results['img'] = DataContainer(results['img'], padding_value=0, stack=True)
        except Exception as e:
            print(e)
            self.error_list.add(idx)
            self.error_list.add(f'{img_id}.jpg')
            print(self.error_list)
            if len(self.error_list) > 20:
                return None
            if not self.test_mode:
                results = self.__getitem__(np.random.randint(0, len(self)))

        return results

    def __getitem__(self, idx):
        results = self.get_data(idx)
        return results

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    def results2json(self, results, outfile_prefix):
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
            json_results = self._det2json(results)
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

    def _proposal2json(self, results):
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids)):
            # ann_ids = self.coco.get_ann_ids(img_ids=self.img_ids[i])
            # ann_info = self.coco.load_anns(ann_ids)
            img_id = self.img_ids[i]
            bboxes_tmp = [[instance["x"], instance["y"], instance["w"], instance["h"]] for instance in self.img_instances_pair[img_id]]
            if len(bboxes_tmp) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in bboxes_tmp:
                x1, y1, w, h = ann
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        # for i in np.random.choice(range(len(self.img_ids)), 100):
        # # for i in range(len(self.img_ids)):
        #     img_id = self.img_ids[i]
        #     filename = os.path.abspath(self.data_root) + '/VG/VG_100K' + f'/{img_id}.jpg'
        #     img = cv2.imread(filename, cv2.IMREAD_COLOR)
        #     for box in gt_bboxes[i]:
        #         x1, y1, x2, y2 = box.astype(np.int)
        #         img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=1)
        #     for box in results[i]:
        #         x1, y1, x2, y2, _ = box.astype(np.int)
        #         img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=1)
        #     os.makedirs('results/tmp', exist_ok=True)
        #     cv2.imwrite('results/tmp' + f'/{img_id}.jpg', img)

        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def evaluate_det_segm(self,
                          results,
                          result_files,
                          coco_gt,
                          metrics,
                          logger=None,
                          classwise=False,
                          proposal_nums=(100, 300, 1000),
                          iou_thrs=None,
                          metric_items=None
                          ):
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
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            assert len(proposal_nums) == 3
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # cocoEval.params.areaRng = areaRng

            # mapping of cocoEval.stats
            # my settings
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                f'AR_{proposal_nums[0]}': 6,
                f'AR_{proposal_nums[1]}': 7,
                f'AR_{proposal_nums[2]}': 8,
                f'AR_s_{proposal_nums[2]}': 9,
                f'AR_m_{proposal_nums[2]}': 10,
                f'AR_l_{proposal_nums[2]}': 11,
                f'AP_50_s': 12,
                f'AP_50_m': 13,
                f'AR_50_s': 14,
                f'AR_50_m': 15,
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

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

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

                if metric_items is None:
                    # metric_items = [
                    #     'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    # ]
                    # my changes
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l',
                        f'AP_50_s', f'AP_50_m', 'AR_50_s', f'AR_50_m'
                    ]

                for metric_item in metric_items:
                    # my changes
                    key = f'{val_or_test}_{metric}_{metric_item}'
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                    )
                    eval_results[key] = val

                # ap = cocoEval.stats[:6]
                # eval_results[f'{metric}_mAP_copypaste'] = (
                #     f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                #     f'{ap[4]:.3f} {ap[5]:.3f}')

                all_metrics = cocoEval.stats
                eval_results[f'{val_or_test}_{metric}_all_metrics'] = ' '.join([f'{x:.3f}' for x in all_metrics])

        return eval_results

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None
                 ):
        pass
