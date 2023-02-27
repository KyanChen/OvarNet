import json

from textblob.wordnet import NOUN
from textblob import Word

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

category_synsets = {}
for cat in cats:
       synonyms = []
       for syn in Word(cat).get_synsets(pos=NOUN):
              for lm in syn.lemmas():
                     synonyms.append(lm.name())
       category_synsets[cat] = list(set(synonyms))
json.dump(category_synsets, open('category_synsets.json', 'w'), indent=4)