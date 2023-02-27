import json
import copy
import re
from textblob import TextBlob
from tqdm import tqdm
from textblob.wordnet import NOUN

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
synset2cocoid = {x['synset']: x['coco_cat_id'] for x in COCO_SYNSET_CATEGORIES}
coco_id2cate = json.load(open('coco_id2cate.json'))

parent_folder = '../../data/COCO/annotations'
# parent_folder = '/Users/kyanchen/Documents/COCO/annotations'
json_file = parent_folder+'/captions_train2017.json'
json_data = json.load(open(json_file, 'r'))
caption_anns = json_data['annotations']
extracted_data = {}
for ann in caption_anns:
    image_id = ann['image_id']
    caption = ann['caption'].strip()
    if caption[-1] not in ['.', '。', '?', '!', ';']:
        caption += '.'
    extracted_data[image_id] = extracted_data.get(image_id, {})
    extracted_data[image_id]['caption'] = extracted_data[image_id].get('caption', []) + [caption]

categories = json.load(open('common2common_category2id_48_32.json', 'r'))
categories = list(categories['common1'].keys()) + list(categories['common2'].keys())
print('len category: ', len(categories))

atts = json.load(open('../VAW/common2rare_att2id.json', 'r'))
atts = list(atts['common'].keys()) + list(atts['rare'].keys())
print('len att: ', len(atts))

def punc_filter(text):
    # rule = re.compile(r'[^\-a-zA-Z0-9]')
    rule = re.compile('^\w+$')  # 由数字、26个英文字母或者下划线组成的字符串
    text = rule.sub(' ', text)
    text = ' '.join([x.strip() for x in text.split(' ') if len(x.strip()) > 0])
    return text

extracted_data_tmp = extracted_data.copy()
for img_id, item in tqdm(extracted_data_tmp.items()):
    extracted_data[img_id]['category'] = []
    extracted_data[img_id]['attribute'] = []
    captions = item['caption']
    # for caption in captions:
    caption = ' '.join(captions)
    caption = punc_filter(caption)
    caption = caption.lower()
    for category in categories:
        rex = re.search(rf'\b{category}\b', caption)
        if rex is not None:
            extracted_data[img_id]['category'] += [category]
    for att in atts:
        rex = re.search(rf'\b{att}\b', caption)
        if rex is not None:
            extracted_data[img_id]['attribute'] += [att]

    speech = TextBlob(caption)
    for word, tag in speech.pos_tags:
        if tag in ['NN', 'NNS']:
            if tag == 'NNS':
                word = word.singularize()
            for syn in word.get_synsets(pos=NOUN):
                synset2cocoid_keys = list(synset2cocoid.keys())
                syn = syn.name()
                if syn in synset2cocoid_keys:
                    extracted_data[img_id]['category'] += [coco_id2cate[str(synset2cocoid[syn])]]
    noun_phrases = [str(x) for x in speech.noun_phrases]
    # extracted_data[img_id]['phase'] += noun_phrases
    extracted_data[img_id]['phase'] = list(set(noun_phrases))
    extracted_data[img_id]['category'] = list(set(extracted_data[img_id]['category']))
    extracted_data[img_id]['attribute'] = list(set(extracted_data[img_id]['attribute']))

json.dump(extracted_data, open(parent_folder+'/train_2017_caption_tagging.json', 'w'), indent=4)
