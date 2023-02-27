import glob
import json
import os
import cv2
import matplotlib.pyplot as plt
import tqdm
from imagesize import imagesize
import random

data_root = '/Users/kyanchen/Documents/LSA/vis_test/tmp'
dataset2path = {
    'vg': 'VG/VG_100K',
    'flickr': 'flickr/images',
    'coco': 'COCO/COCOall',
    'oi': 'OpenImages'
}
# data = glob.glob(data_root+'/*.jpg')
for idx in tqdm.tqdm(range(36, 100)):
    img_file = data_root + f'/{idx}.jpg'
    objs = json.load(open(img_file.replace('.jpg', '.json')))
    img_id = objs['image_id']
    dataset_name, img_file_name = img_id.split('_')
    img_w, img_h = imagesize.get(img_file)
    objects = objs['objects']
    img = cv2.imread(img_file)
    for i_object in objects:
        img_c = img.copy()
        x1, y1, x2, y2 = i_object['box']
        print(i_object)
        # if i_object.get('ground', '') == 'mousetrace':
        #     continue
        if x1 == -1 and y1 == -1 and x2 == -1 and y2 == -1:  # 全图，有"instance_id" "ground": "none", COCO
            l, t, r, b = [0, 0, img_w - 1, img_h - 1]
        elif 0 <= x1 <= 1 and 0 <= x2 <= 1 and 0 <= y1 <= 1 and 0 <= y2 <= 1:  # "ground": "mousetrace", COCO
            # cx, cy, obj_w, obj_h = x1*img_w, y1*img_h, x2*img_w, y2*img_h
            # l, t, r, b = cx-obj_w/2, cy-obj_h/2, cx+obj_w/2, cy+obj_h/2
            l, t, r, b = x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h
        elif dataset_name == 'vg' and x2 > 1 and y2 > 1:  # 有"instance_id" "ground": "box" VG
            # cx, cy, obj_w, obj_h = x1*img_w, y1*img_h, x2*img_w, y2*img_h
            # l, t, r, b = cx-obj_w/2, cy-obj_h/2, cx+obj_w/2, cy+obj_h/2
            l, t, r, b = x1, y1, x1+x2, y1+y2
            # l, t, r, b = x1, y1, x2, y2
        elif dataset_name == 'flickr' and x2 > 1 and y2 > 1:  # 有"instance_id" "ground": "box","none" flickr
            l, t, r, b = x1, y1, x2, y2
        elif dataset_name == 'coco' and x2 > 1 and y2 > 1:  # 有"instance_id" "ground": "box","none" flickr
            l, t, r, b = x1, y1, x2, y2
        else:
            print('eeeee')

        img_c = cv2.rectangle(img_c, (int(l), int(t)), (int(r), int(b)), (0, 0, 255), thickness=2)
        print(f'{i_object["object"]} {i_object["box"]}')
        cv2.imshow(f'{idx}', img_c)
        cv2.waitKey(0)

