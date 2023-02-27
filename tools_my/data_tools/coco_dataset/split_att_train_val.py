import json
import os.path

common1 = [
    "toilet",
    "bicycle",
    "apple",
    "train",
    "laptop",
    "carrot",
    "motorcycle",
    "oven",
    "chair",
    "mouse",
    "boat",
    "kite",
    "sheep",
    "horse",
    "sandwich",
    "clock",
    "tv",
    "backpack",
    "toaster",
    "bowl",
    "microwave",
    "bench",
    "book",
    "orange",
    "bird",
    "pizza",
    "fork",
    "frisbee",
    "bear",
    "vase",
    "toothbrush",
    "spoon",
    "giraffe",
    "handbag",
    "broccoli",
    "refrigerator",
    "remote",
    "surfboard",
    "car",
    "bed",
    "banana",
    "donut",
    "skis",
    "person",
    "truck",
    "bottle",
    "suitcase",
    "zebra",
    # "background"
]

common2 = [
    "umbrella",
    "cow",
    "cup",
    "bus",
    "keyboard",
    "skateboard",
    "dog",
    "couch",
    "tie",
    "snowboard",
    "sink",
    "elephant",
    "cake",
    "scissors",
    "airplane",
    "cat",
    "knife"]
save_path = '../../../attributes/COCO'
common2common = {'common1': {}, 'common2': {}}
for idx, item in enumerate(common1):
    common2common['common1'].update({item: idx})
common2common['common1_len'] = len(common2common['common1'])
for idx, item in enumerate(common2):
    common2common['common2'].update({item: idx+common2common['common1_len']})
common2common['common2_len'] = len(common2common['common2'])

json.dump(common2common, open(save_path+'/common2common_category2id_48_17.json', 'w'), indent=4)
