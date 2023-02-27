import json
import math
import multiprocessing
import os
import argparse
# import img2dataset
import re

import pandas as pd
from textblob import TextBlob
from textblob.wordnet import NOUN
from tqdm import tqdm
from concurrent import futures

COCO_SYNSET_CATEGORIES = [
    {"synset": "person.n.01", "coco_cat_id": 1},
    {"synset": "bicycle.n.01", "coco_cat_id": 2},
    {"synset": "car.n.01", "coco_cat_id": 3},
    {"synset": "motorcycle.n.01", "coco_cat_id": 4},
    {"synset": "airplane.n.01", "coco_cat_id": 5},
    {"synset": "bus.n.01", "coco_cat_id": 6},
    {"synset": "train.n.01", "coco_cat_id": 7},
    {"synset": "truck.n.01", "coco_cat_id": 8},
    {"synset": "boat.n.01", "coco_cat_id": 9},
    {"synset": "traffic_light.n.01", "coco_cat_id": 10},
    {"synset": "fireplug.n.01", "coco_cat_id": 11},
    {"synset": "stop_sign.n.01", "coco_cat_id": 13},
    {"synset": "parking_meter.n.01", "coco_cat_id": 14},
    {"synset": "bench.n.01", "coco_cat_id": 15},
    {"synset": "bird.n.01", "coco_cat_id": 16},
    {"synset": "cat.n.01", "coco_cat_id": 17},
    {"synset": "dog.n.01", "coco_cat_id": 18},
    {"synset": "horse.n.01", "coco_cat_id": 19},
    {"synset": "sheep.n.01", "coco_cat_id": 20},
    {"synset": "beef.n.01", "coco_cat_id": 21},
    {"synset": "elephant.n.01", "coco_cat_id": 22},
    {"synset": "bear.n.01", "coco_cat_id": 23},
    {"synset": "zebra.n.01", "coco_cat_id": 24},
    {"synset": "giraffe.n.01", "coco_cat_id": 25},
    {"synset": "backpack.n.01", "coco_cat_id": 27},
    {"synset": "umbrella.n.01", "coco_cat_id": 28},
    {"synset": "bag.n.04", "coco_cat_id": 31},
    {"synset": "necktie.n.01", "coco_cat_id": 32},
    {"synset": "bag.n.06", "coco_cat_id": 33},
    {"synset": "frisbee.n.01", "coco_cat_id": 34},
    {"synset": "ski.n.01", "coco_cat_id": 35},
    {"synset": "snowboard.n.01", "coco_cat_id": 36},
    {"synset": "ball.n.06", "coco_cat_id": 37},
    {"synset": "kite.n.03", "coco_cat_id": 38},
    {"synset": "baseball_bat.n.01", "coco_cat_id": 39},
    {"synset": "baseball_glove.n.01", "coco_cat_id": 40},
    {"synset": "skateboard.n.01", "coco_cat_id": 41},
    {"synset": "surfboard.n.01", "coco_cat_id": 42},
    {"synset": "tennis_racket.n.01", "coco_cat_id": 43},
    {"synset": "bottle.n.01", "coco_cat_id": 44},
    {"synset": "wineglass.n.01", "coco_cat_id": 46},
    {"synset": "cup.n.01", "coco_cat_id": 47},
    {"synset": "fork.n.01", "coco_cat_id": 48},
    {"synset": "knife.n.01", "coco_cat_id": 49},
    {"synset": "spoon.n.01", "coco_cat_id": 50},
    {"synset": "bowl.n.03", "coco_cat_id": 51},
    {"synset": "banana.n.02", "coco_cat_id": 52},
    {"synset": "apple.n.01", "coco_cat_id": 53},
    {"synset": "sandwich.n.01", "coco_cat_id": 54},
    {"synset": "orange.n.01", "coco_cat_id": 55},
    {"synset": "broccoli.n.01", "coco_cat_id": 56},
    {"synset": "carrot.n.01", "coco_cat_id": 57},
    # {"synset": "frank.n.02", "coco_cat_id": 58},
    {"synset": "sausage.n.01", "coco_cat_id": 58},
    {"synset": "pizza.n.01", "coco_cat_id": 59},
    {"synset": "doughnut.n.02", "coco_cat_id": 60},
    {"synset": "cake.n.03", "coco_cat_id": 61},
    {"synset": "chair.n.01", "coco_cat_id": 62},
    {"synset": "sofa.n.01", "coco_cat_id": 63},
    {"synset": "pot.n.04", "coco_cat_id": 64},
    {"synset": "bed.n.01", "coco_cat_id": 65},
    {"synset": "dining_table.n.01", "coco_cat_id": 67},
    {"synset": "toilet.n.02", "coco_cat_id": 70},
    {"synset": "television_receiver.n.01", "coco_cat_id": 72},
    {"synset": "laptop.n.01", "coco_cat_id": 73},
    {"synset": "mouse.n.04", "coco_cat_id": 74},
    {"synset": "remote_control.n.01", "coco_cat_id": 75},
    {"synset": "computer_keyboard.n.01", "coco_cat_id": 76},
    {"synset": "cellular_telephone.n.01", "coco_cat_id": 77},
    {"synset": "microwave.n.02", "coco_cat_id": 78},
    {"synset": "oven.n.01", "coco_cat_id": 79},
    {"synset": "toaster.n.02", "coco_cat_id": 80},
    {"synset": "sink.n.01", "coco_cat_id": 81},
    {"synset": "electric_refrigerator.n.01", "coco_cat_id": 82},
    {"synset": "book.n.01", "coco_cat_id": 84},
    {"synset": "clock.n.01", "coco_cat_id": 85},
    {"synset": "vase.n.01", "coco_cat_id": 86},
    {"synset": "scissors.n.01", "coco_cat_id": 87},
    {"synset": "teddy.n.01", "coco_cat_id": 88},
    {"synset": "hand_blower.n.01", "coco_cat_id": 89},
    {"synset": "toothbrush.n.01", "coco_cat_id": 90},
]

synset2cocoid = {x['synset']: x['coco_cat_id'] for x in COCO_SYNSET_CATEGORIES}
coco_id2cate = json.load(open('../COCO/coco_id2cate.json'))

categories = json.load(open('../COCO/common2common_category2id_48_32.json', 'r'))
categories = list(categories['common1'].keys()) + list(categories['common2'].keys())
print('len category: ', len(categories))

atts = json.load(open('../VAW/common2rare_att2id.json', 'r'))
atts = list(atts['common'].keys()) + list(atts['rare'].keys())
print('len att: ', len(atts))

all_atts = categories + atts

def punc_filter(text):
    # rule = re.compile(r'[^\-a-zA-Z0-9]')
    rule = re.compile('^\w+$')  # 由数字、26个英文字母或者下划线组成的字符串
    text = rule.sub(' ', text)
    text = ' '.join([x.strip() for x in text.split(' ') if len(x.strip()) > 0])
    return text


def dict_slice(dict0, start, end):
    _dict = dict0
    keys = list(_dict.keys())
    _dict_slice = {}
    for key in keys[start: end]:   # 通过index方法，让列表自己找到索引值并返回
        _dict_slice[key] = _dict[key]
    return _dict_slice


def split_json_data(text_dict, split_num, path):
    os.makedirs(path, exist_ok=True)
    n_item_per_slice = len(text_dict) // split_num
    for i in range(split_num):
        start = i * n_item_per_slice
        end = min(start + n_item_per_slice, len(text_dict))
        json.dump(dict_slice(text_dict, start, end), open(path+f'/split_{i}.json', 'w'))

def tagging_cc(pid, keys, save_folder):
    for img_key in tqdm(keys):
        sub_folder = img_key[:5]
        json_data = json.load(open(save_folder + f'/../CC3MImages/{sub_folder}/{img_key}.json', 'r'))
        json_data['category'] = []
        json_data['attribute'] = []
        caption = json_data['caption']
        caption = punc_filter(caption)
        caption = caption.lower()
        for category in categories:
            rex = re.search(rf'\b{category}\b', caption)
            if rex is not None:
                json_data['category'] += [category]
        for att in atts:
            rex = re.search(rf'\b{att}\b', caption)
            if rex is not None:
                json_data['attribute'] += [att]

        speech = TextBlob(caption)
        for word, tag in speech.pos_tags:
            if tag in ['NN', 'NNS']:
                if tag == 'NNS':
                    word = word.singularize()
                for syn in word.get_synsets(pos=NOUN):
                    synset2cocoid_keys = list(synset2cocoid.keys())
                    syn = syn.name()
                    if syn in synset2cocoid_keys:
                        json_data['category'] += [coco_id2cate[str(synset2cocoid[syn])]]
        noun_phrases = [str(x) for x in speech.noun_phrases]
        json_data['phase'] = list(set(noun_phrases))
        json_data['category'] = list(set(json_data['category']))
        json_data['attribute'] = list(set(json_data['attribute']))
        json.dump(json_data, open(save_folder + f'/{img_key}.json', 'w'), indent=4)


def gather_all(path, split_num):
    return_data = {}
    for i in range(split_num):
        data = json.load(open(path + f'/split_{i}_filtered.json', 'r'))
        return_data.update(data)
    return return_data


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    n_process = 128

    folder = '../../data/CC3M'
    save_folder = folder+'/tagging_labels'
    os.makedirs(save_folder, exist_ok=True)
    train_cc_keys = json.load(open('train_CC_keys.json', 'r'))
    # 001535015
    split_keys = []
    n_item_per_slice = math.ceil(len(train_cc_keys) / n_process)
    for i in range(n_process):
        start = i * n_item_per_slice
        end = min(start + n_item_per_slice, len(train_cc_keys))
        split_keys.append(train_cc_keys[start:end])

    process_list = []
    for pid in range(n_process):
        print('pid {}'.format(pid))
        process_list.append(
            multiprocessing.Process(target=tagging_cc, args=(pid, split_keys[pid], save_folder))
        )
    [p.start() for p in process_list]
    [p.join() for p in process_list]
