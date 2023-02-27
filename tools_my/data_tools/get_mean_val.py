import argparse
import os.path as osp

import mmcv
import numpy as np
import json
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.utils import get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Optimize anchor parameters.')
    parser.add_argument('--config', default='../../configs_my/detr_rs.py', help='Train config file path.')
    parser.add_argument(
        '--output-dir',
        default='./img_mean_stds',
        type=str,
        help='Path to save anchor optimize result.')

    args = parser.parse_args()
    return args


def get_img_mean_std(dataset, logger):
    logger.info('Collecting img from dataset...')
    means = []
    stds = []
    prog_bar = mmcv.ProgressBar(len(dataset))
    for item in dataset.data_infos:
        prog_bar.update()
        img_path = dataset.img_prefix + '/' + item['file_name']
        img = mmcv.imread(img_path, channel_order='rgb').astype(np.uint8)
        assert img is not None, img_path + 'is not valid'
        # height*width*channels, axis=0 is the first dim
        mean = np.mean(np.mean(img, axis=0), axis=0)
        means.append(mean)
        std = np.std(np.std(img, axis=0), axis=0)
        stds.append(std)
    mean = np.mean(np.array(means), axis=0).tolist()
    std = np.mean(np.array(stds), axis=0).tolist()
    # BGR
    return {'mean': mean, 'std': std}


def main():
    logger = get_root_logger()
    args = parse_args()
    cfg = args.config
    cfg = Config.fromfile(cfg)

    train_data_cfg = cfg.data.train
    while 'dataset' in train_data_cfg:
        train_data_cfg = train_data_cfg['dataset']
    dataset = build_dataset(train_data_cfg)

    info = get_img_mean_std(dataset=dataset, logger=logger)
    writer = osp.join(args.output_dir, 'mean_std_info.json')
    mmcv.mkdir_or_exist(osp.dirname(writer))
    with open(writer, 'w') as f_writer:
        json.dump(info, f_writer)
    logger.info('\nmean=%s, std=%s\n' % (info['mean'], info['std']))


if __name__ == '__main__':
    main()
