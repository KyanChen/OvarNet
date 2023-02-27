import json
import pickle

cats = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
       'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
       'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
       'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
       'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
       'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
       'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
       'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
       'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
       'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
       'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
       'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
       'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

category_map = {'num_categories': 0, 'num_sub_categories': 0, 'id2category': {}, 'category2id': {}}
category_map['id2category'] = {k: v for k, v in enumerate(cats)}
for k, v in enumerate(cats):
    category_map['category2id'][v] = k
category_map['num_categories'] = len(category_map['id2category'])
category_map['num_sub_categories'] = len(category_map['category2id'])
json.dump(category_map, open(r'category_id_map.json', 'w'), indent=4)