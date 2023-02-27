import json
import os
import numpy as np
import mmcv
import tqdm
import multiprocessing
import re

def dict_slice(dict0, start, end):
    _dict = dict0
    keys = list(_dict.keys())
    _dict_slice = {}
    for key in keys[start: end]:   # 通过index方法，让列表自己找到索引值并返回
        _dict_slice[key] = _dict[key]
    return _dict_slice

def get_category_attribute_pair(all_pair, captions: dict, category_id_map, attribute_id_map, pid):
    def punc_filter(text):
        rule = re.compile(r'[^\-a-zA-Z0-9]')
        text = rule.sub(' ', text)
        text = ' '.join([x.strip() for x in text.split(' ') if len(x.strip()) > 0])
        return text

    for img_id, caption_data in tqdm.tqdm(captions.items()):
        labels = {}
        for caption in caption_data:
            categories = []
            categories_index = []
            attributes = []
            attributes_index = []

            caption = punc_filter(caption)
            caption = caption.lower()
            for category, id in category_id_map['category2id'].items():
                rule = re.compile(rf'\b{category}\b')
                caption = rule.sub(f'c{id}c', caption)
            for attribute, id in attribute_id_map['attribute2id'].items():
                rule = re.compile(rf'\b{attribute}\b')
                caption = rule.sub(f'a{id}a', caption)
            try:
                for idx, char_x in enumerate(caption.split(' ')):
                    char_x_tmp = re.search(r'\bc\d{1,}c\b', char_x)
                    if char_x_tmp is not None:
                        categories.append(int(char_x_tmp.group().replace('c', '')))
                        categories_index.append(idx)
                    char_x_tmp = re.search(r'\ba\d{1,}a\b', char_x)
                    if char_x_tmp is not None:
                        attributes.append(int(char_x_tmp.group().replace('a', '')))
                        attributes_index.append(idx)
            except:
                print(caption)
            # assign
            categories_index = np.array(categories_index).reshape(1, -1)
            attributes_index = np.array(attributes_index).reshape(-1, 1)
            dis = np.abs(attributes_index - categories_index)
            if np.size(dis) == 0:
                continue
            assigned_categories = np.argmin(dis, axis=1)  # att belong to which category
            for idx, assigned_category in enumerate(assigned_categories):
                labels[categories[assigned_category]] = labels.get(categories[assigned_category], []) + [attributes[idx]]
        returned_labels = {}
        for k, v in labels.items():
            if len(v) > 0:
                returned_labels[k] = list(set(v))
        all_pair[img_id] = returned_labels


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    n_process = 8
    category_id_map = '../../objects/MSCOCO/category_id_map.json'
    attribute_id_map = '../../attributes/COCO/attribute_id_map.json'

    caption_data = mmcv.load('captions_val2014.json')
    category_id_map = mmcv.load(category_id_map)
    attribute_id_map = mmcv.load(attribute_id_map)
    caption = {}
    for ann in caption_data['annotations']:
        caption[ann['image_id']] = caption.get(ann['image_id'], []) + [ann['caption']]

    data_slice_list = []
    n_item_per_slice = int(len(caption) / n_process)
    for i in range(n_process):
        start = i * n_item_per_slice
        end = min(start + n_item_per_slice, len(caption))
        data_slice_list.append(dict_slice(caption, start, end))

    all_pair = multiprocessing.Manager().dict()
    process_list = []
    for pid in range(n_process):
        print('pid {}'.format(pid))
        process_list.append(
            multiprocessing.Process(target=get_category_attribute_pair, args=(all_pair, data_slice_list[pid], category_id_map, attribute_id_map, pid))
        )
    [p.start() for p in process_list]
    [p.join() for p in process_list]

    json.dump(dict(all_pair), open(f'category_attribute_pairs.json', 'w'), indent=4)
