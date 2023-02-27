import json
import os

import tqdm
from textblob import TextBlob
from textblob import Word
from textblob.taggers import NLTKTagger
import multiprocessing

nltk_tagger = NLTKTagger()

# https://github.com/lvis-dataset/lvis-api/blob/master/data/coco_to_synset.json
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

def get_att_categories(pid, path):
    data = json.load(open(path + f'/split_{pid}.json', 'r'))
    return_data = {'atts': {}, 'categories': {}}
    for text_str in tqdm.tqdm(data):
        blob = TextBlob(text_str, pos_tagger=nltk_tagger)
        # print(blob.pos_tags)
        for word, tag in blob.pos_tags:
            if tag == 'JJ':
                return_data['atts'][word] = return_data['atts'].get(word, 0) + 1
            elif tag == 'NN':
                return_data['categories'][word] = return_data['categories'].get(word, 0) + 1
            elif tag == 'NNS':
                word = Word(word).singularize()
                return_data['categories'][word] = return_data['categories'].get(word, 0) + 1

    json.dump(return_data, open(path + f'/split_{pid}_atts_categories.json', 'w'), indent=4)


def split_json_data(text_list, split_num, path):
    os.makedirs(path, exist_ok=True)
    n_item_per_slice = len(text_list) // split_num
    for i in range(split_num):
        start = i * n_item_per_slice
        end = min(start + n_item_per_slice, len(json_data))
        json.dump(text_list[start: end], open(path+f'/split_{i}.json', 'w'))


def gather_all(path, split_num):
    return_data = {'num_atts': 0, 'num_categories': 0, 'atts': {}, 'categories': {}}
    for i in range(split_num):
        data = json.load(open(path + f'/split_{i}_atts_categories.json', 'r'))
        for k, v in data['atts'].items():
            return_data['atts'][k] = return_data['atts'].get(k, 0) + v
        for k, v in data['categories'].items():
            return_data['categories'][k] = return_data['categories'].get(k, 0) + v

    return_data['atts'] = sorted(return_data['atts'].items(), key=lambda kv: kv[1], reverse=True)
    return_data['categories'] = sorted(return_data['categories'].items(), key=lambda kv: kv[1], reverse=True)
    return_data['num_atts'] = len(return_data['atts'])
    return_data['num_categories'] = len(return_data['categories'])
    return return_data


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    n_process = 32

    json_data = json.load(open('caption_all/caption_seg_word.json', 'r'))['captions']
    split_json_data(json_data, split_num=n_process, path='caption_all/tmp')

    process_list = []
    for pid in range(n_process):
        print('pid {}'.format(pid))
        process_list.append(
            multiprocessing.Process(target=get_att_categories, args=(pid, 'caption_all/tmp'))
        )
    [p.start() for p in process_list]
    [p.join() for p in process_list]

    return_data = gather_all(path='caption_all/tmp', split_num=n_process)
    json.dump(return_data, open(f'caption_all/extracted_atts_categories.json', 'w'), indent=4)
