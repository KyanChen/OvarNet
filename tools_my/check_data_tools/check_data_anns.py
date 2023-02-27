import json
import os
import cv2
import tqdm
from imagesize import imagesize
import random

data_root = '/expand_data/datasets'
json_data = json.load(open(data_root + '/LSA/' + f'test.json', 'r'))
dataset2path = {
    'vg': 'VG/VG_100K',
    'flickr': 'flickr/images',
    'coco': 'COCO/COCOall',
    'oi': 'OpenImages'
}
random.shuffle(json_data)
os.makedirs('results/tmp', exist_ok=True)
for idx, img_item in tqdm.tqdm(enumerate(json_data[:100])):
    img_id = img_item['image_id']
    dataset_name, img_file_name = img_id.split('_')
    if dataset_name == 'coco':
        img_file_name = f'{int(img_file_name):012d}'
    img_info = {'file_name': f'{img_file_name + ".jpg"}'}
    img_path = os.path.abspath(data_root) + f'/{dataset2path[dataset_name]}/' + img_info['file_name']
    w, h = imagesize.get(img_path)
    img_info['width'] = w
    img_info['height'] = h
    objects = img_item['objects']
    img = cv2.imread(img_path)
    for i_object in objects:
        x, y, w, h = i_object['box']
        if x < 0 and y < 0:
            x, y, w, h = [0, 0, img_info['width'] - 1, img_info['height'] - 1]
        elif x <= 1 and y <= 1 and w <= 1 and h <= 1:
            x, y, w, h = [
                x * img_info['width'], y * img_info['height'],
                w * img_info['width'], h * img_info['height']]
        if dataset_name == 'vg':
            crop_box = [x, y, x + w, y + h]
        elif dataset_name == 'coco':
            crop_box = [x, y, x + w, y + h]
        elif dataset_name == 'flickr':
            crop_box = [x, y, w, h]
        elif dataset_name == 'oi':
            crop_box = [x, y, w, h]
        crop_box = list(map(int, crop_box))
        # img = cv2.rectangle(img, crop_box[0:2], crop_box[2:4], (0, 0, 255), thickness=2)
    json.dump(img_item, open('results/tmp' + f'/{idx}.json', 'w'), indent=4)
    cv2.imwrite('results/tmp' + f'/{idx}.jpg', img)

