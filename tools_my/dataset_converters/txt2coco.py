import argparse
import glob
import os
import os.path as osp
import xml.etree.ElementTree as ET

import cv2
import mmcv
import numpy as np

class_name = [
    'airplane', 'ship', 'storage tank', 'baseball diamond',
    'tennis court', 'basketball court', 'ground track field',
    'harbor', 'bridge', 'vehicle'
]

label_ids = {name: i for i, name in enumerate(class_name)}


def txt_paraser(txt_name):
    """
    class_name left top right bottom or none
    :param txt_name: [class_ID, left, right, top, bottom]
    :return: narray, num_objrcts * 5
    """

    def convert_func(x):
        return float(x.decode().replace('(', '').replace(')', ''))

    def get_label_id(x):
        # return label_ids[x.decode()]

        return int(x.decode())-1

    targets = np.loadtxt(txt_name,
                         float,
                         ndmin=2,
                         delimiter=',',
                         converters={
                             0: convert_func,
                             1: convert_func,
                             2: convert_func,
                             3: convert_func,
                             4: get_label_id
                         }
                         )
    return targets


def parse_txt(args):
    txt_path, img_path, parent_path = args
    img = mmcv.imread(img_path)
    h, w, c = img.shape

    if osp.exists(txt_path) and osp.getsize(txt_path) > 2:
        targets = txt_paraser(txt_path)
        bboxes = targets[:, :-1]
        # bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, :2]
        bboxes[:, ::2] = np.clip(bboxes[:, ::2], a_min=0, a_max=w)
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], a_min=0, a_max=h)
        labels = targets[:, -1]
    else:
        bboxes = np.zeros((0, 4))
        labels = np.zeros((0, ))
    bboxes_ignore = np.zeros((0, 4))
    labels_ignore = np.zeros((0, ))

    annotation = {
        'filename': img_path.replace(parent_path+os.sep, '').replace('\\', '/'),
        'width': w,
        'height': h,
        'ann': {
            'bboxes': bboxes.astype(np.float32),
            'labels': labels.astype(np.int64),
            'bboxes_ignore': bboxes_ignore.astype(np.float32),
            'labels_ignore': labels_ignore.astype(np.int64)
        }
    }
    return annotation


def _get_file(in_path_list):
    file_list = []
    for file in in_path_list:
        if osp.isdir(file):
            files = glob.glob(file + '/*')
            file_list.extend(_get_file(files))
        else:
            file_list += [file]
    return file_list


def cvt_annotations(devkit_path, out_file, img_format_list=['jpg', 'tif', 'tiff', 'png']):
    annotations = []
    filelist = _get_file([devkit_path])
    img_paths = [x for x in filelist if x.split('.')[-1] in img_format_list]
    label_paths = [x.replace('tiff', 'txt').replace('jpg', 'txt').replace('png', 'txt') for x in img_paths]
    parent_paths = [devkit_path] * len(img_paths)
    part_annotations = mmcv.track_progress(parse_txt, list(zip(label_paths, img_paths, parent_paths)))
    annotations.extend(part_annotations)

    if out_file.endswith('json'):
        annotations = cvt_to_coco_json(annotations)
    mmcv.dump(annotations, out_file, indent=4)
    return annotations


def cvt_to_coco_json(annotations):
    image_id = 0
    annotation_id = 0
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instance'
    coco['categories'] = []
    coco['annotations'] = []
    image_set = set()

    def addAnnItem(annotation_id, image_id, category_id, bbox, difficult_flag):
        annotation_item = dict()
        annotation_item['segmentation'] = []

        seg = []
        # bbox[] is x1,y1,x2,y2
        # left_top
        seg.append(int(bbox[0]))
        seg.append(int(bbox[1]))
        # left_bottom
        seg.append(int(bbox[0]))
        seg.append(int(bbox[3]))
        # right_bottom
        seg.append(int(bbox[2]))
        seg.append(int(bbox[3]))
        # right_top
        seg.append(int(bbox[2]))
        seg.append(int(bbox[1]))

        annotation_item['segmentation'].append(seg)

        xywh = np.array(
            [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
        annotation_item['area'] = int(xywh[2] * xywh[3])
        if difficult_flag == 1:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 1
        else:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 0
        annotation_item['image_id'] = int(image_id)
        annotation_item['bbox'] = xywh.astype(int).tolist()
        annotation_item['category_id'] = int(category_id)
        annotation_item['id'] = int(annotation_id)
        coco['annotations'].append(annotation_item)
        return annotation_id + 1

    for category_id, name in enumerate(class_name):
        category_item = dict()
        category_item['supercategory'] = str('none')
        category_item['id'] = int(category_id)
        category_item['name'] = str(name)
        coco['categories'].append(category_item)

    for ann_dict in annotations:
        file_name = ann_dict['filename']
        ann = ann_dict['ann']
        assert file_name not in image_set
        image_item = dict()
        image_item['id'] = int(image_id)
        image_item['file_name'] = str(file_name)
        image_item['height'] = int(ann_dict['height'])
        image_item['width'] = int(ann_dict['width'])
        coco['images'].append(image_item)
        image_set.add(file_name)

        bboxes = ann['bboxes'][:, :4]
        labels = ann['labels']
        for bbox_id in range(len(bboxes)):
            bbox = bboxes[bbox_id]
            label = labels[bbox_id]
            annotation_id = addAnnItem(
                annotation_id, image_id, label, bbox, difficult_flag=0)

        bboxes_ignore = ann['bboxes_ignore'][:, :4]
        labels_ignore = ann['labels_ignore']
        for bbox_id in range(len(bboxes_ignore)):
            bbox = bboxes_ignore[bbox_id]
            label = labels_ignore[bbox_id]
            annotation_id = addAnnItem(
                annotation_id, image_id, label, bbox, difficult_flag=1)

        image_id += 1

    return coco


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert PASCAL VOC annotations to mmdetection format')
    parser.add_argument('--devkit_path', default=r'D:\Dataset\NWPU VHR-10 dataset\PositiveTrain', help='pascal voc devkit path')
    parser.add_argument('-o', '--out-dir', default=r'D:\Dataset\NWPU VHR-10 dataset', help='output path')
    parser.add_argument('--dataset_name', default='NWPU_train', help='dataset name')
    parser.add_argument(
        '--out-format',
        default='coco',
        choices=('pkl', 'coco'),
        help='output format, "coco" indicates coco annotation format')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    devkit_path = args.devkit_path
    out_dir = args.out_dir if args.out_dir else devkit_path+'/..'
    mmcv.mkdir_or_exist(out_dir)

    out_fmt = f'.{args.out_format}'
    if args.out_format == 'coco':
        out_fmt = '.json'

    dataset_name = args.dataset_name
    print(f'processing {dataset_name} ...')
    cvt_annotations(devkit_path, osp.join(out_dir, dataset_name + out_fmt))
    print('Done!')


if __name__ == '__main__':
    main()
