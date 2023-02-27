import json

import torch.utils.data
from mmcv.parallel import DataContainer
from ..datasets.builder import DATASETS
from ..datasets import CocoDataset


# 训练并测试RPN在COCO上的性能
from ..datasets.pipelines import Compose


@DATASETS.register_module()
class RPNOnCocoCapDataset(CocoDataset):
    def prepare_train_img(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        results = self.pipeline(results)
        return results

    def prepare_test_img(self, idx):
        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        results = [x[0].cpu().numpy() for x in results]
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        coco_gt = self.coco
        self.cat_ids = coco_gt.get_cat_ids(cat_names=self.CLASSES)

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        eval_results = self.evaluate_det_segm(results, result_files, coco_gt,
                                              metrics, logger, classwise,
                                              proposal_nums, iou_thrs,
                                              metric_items)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results


@DATASETS.register_module()
class RPNOnCC3MDataset(torch.utils.data.Dataset):
    CLASSES = None
    def __init__(self, keys_file, pipeline, test_mode=True):
        super(RPNOnCC3MDataset, self).__init__()
        self.img_prefix = '/data/kyanchen/prompt/data/CC3M'
        self.keys = json.load(open(keys_file))
        self.pipeline = Compose(pipeline)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        img_key = self.keys[idx]
        sub_folder = img_key[:5]

        results = dict()
        results['img_prefix'] = self.img_prefix+f'/CC3MImages/{sub_folder}'
        results['img_info'] = {}
        results['img_info']['filename'] = f'{img_key}.jpg'
        results['img_key'] = DataContainer(img_key, stack=False, cpu_only=True)
        return self.pipeline(results)

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        results = [x.cpu().numpy() for x in results]
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        coco_gt = self.coco
        self.cat_ids = coco_gt.get_cat_ids(cat_names=self.CLASSES)

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        eval_results = self.evaluate_det_segm(results, result_files, coco_gt,
                                              metrics, logger, classwise,
                                              proposal_nums, iou_thrs,
                                              metric_items)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

