import json
import pickle
import random

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms

file_attr = '../../attributes/VAW/common2common_att2id.json'
file_cate = '../../attributes/COCO/common2common_category2id_48_17.json'

att2id = {}
att_seen_unseen = {}
att2id_data = json.load(open(file_attr, 'r'))
att2id.update(att2id_data['common1'])
att2id.update(att2id_data['common2'])
att_seen_unseen['seen'] = list(att2id_data['common1'].keys())
att_seen_unseen['unseen'] = list(att2id_data['common2'].keys())

category2id = {}
category_seen_unseen = {}
category2id_data = json.load(open(file_cate, 'r'))
category2id.update(category2id_data['common1'])
category2id.update(category2id_data['common2'])
category_seen_unseen['seen'] = list(category2id_data['common1'].keys())
category_seen_unseen['unseen'] = list(category2id_data['common2'].keys())
att2id = {k: v - min(att2id.values()) for k, v in att2id.items()}
category2id = {k: v - min(category2id.values()) for k, v in category2id.items()}
id2att = {v: k for k, v in att2id.items()}
id2category = {v: k for k, v in category2id.items()}

# det_bboxes, keep_idxs = batched_nms(
#                     pred_boxes[mask_pos], pred_scores[mask_pos], pred_label, nms_cfg)
#                 pred_label = pred_label[keep_idxs]
CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
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
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
           (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
           (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
           (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
           (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
           (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
           (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
           (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
           (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
           (134, 134, 103), (145, 148, 174), (255, 208, 186),
           (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
           (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
           (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
           (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
           (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
           (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
           (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
           (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
           (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
           (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
           (191, 162, 208)]

def vis(img, preds, file_name):
    preds = torch.from_numpy(preds)
    boxes = preds[:, :4]
    c_objs = preds[:, 4]
    attr_probs = preds[:, 6:len(att2id)].sigmoid()
    cate_probs = preds[:, -len(category2id):].softmax(dim=-1)

    unseen_cate_id = [category2id[k] for k in category_seen_unseen['unseen']]
    unseen_att_id = [att2id[k] for k in att_seen_unseen['unseen']]

    cate_probs_fuse = torch.sqrt(cate_probs * c_objs[:, None])
    scores, inds = torch.max(cate_probs_fuse, dim=-1)

    mask_pos = scores > 0.6
    boxes = boxes[mask_pos]
    scores = scores[mask_pos]
    pred_label = inds[mask_pos]
    c_objs = c_objs[mask_pos]
    cate_probs = cate_probs[mask_pos]
    attr_probs = attr_probs[mask_pos]

    keep_idxs = batched_nms(boxes, scores, pred_label, iou_threshold=0.5)
    boxes = boxes[keep_idxs]
    pred_label = pred_label[keep_idxs]
    c_objs = c_objs[keep_idxs]
    cate_probs = cate_probs[keep_idxs]
    attr_probs = attr_probs[keep_idxs]

    # values, inds = torch.max(cate_probs, dim=-1)
    filer = open(file_name.replace('.jpg', '.txt'), 'w')
    flag = 0
    rand_select = random.randint(0, 10)
    # rand_select = inds[0]
    for idx, ind in enumerate(pred_label):
        if ind in unseen_cate_id and flag == 0 and cate_probs[idx][ind] > 0.5:
            box = boxes[idx].numpy().astype(int)
            c_obj = c_objs[idx]
            attr_prob = attr_probs[idx]
            cate_prob = cate_probs[idx][ind]
            filer.write(f'object confidence: {c_obj:.2f}\n')
            filer.write('category: \n')
            if ind in unseen_cate_id:
                seen = 'unseen'
            else:
                seen = 'seen'
            filer.write(f'\t{seen} {id2category[ind.item()]} {cate_prob:.2f}\n')
            filer.write('pos attribute: \n')
            att_values, att_inds = torch.sort(attr_prob, descending=True, dim=-1)
            for i_att in range(50):
                if att_inds[i_att] in unseen_att_id:
                    seen = 'unseen'
                else:
                    seen = 'seen'
                filer.write(f'\t{seen} {id2att[att_inds[i_att].item()]} {att_values[i_att]:.2f}\n')
            filer.write('neg attribute: \n')
            for i_att in list(range(len(attr_prob)-3, len(attr_prob)))[::-1]:
                if att_inds[i_att] in unseen_att_id:
                    seen = 'unseen'
                else:
                    seen = 'seen'
                filer.write(f'\t{seen} {id2att[att_inds[i_att].item()]} {att_values[i_att]:.2f}\n')

            img = cv2.rectangle(img, box[:2], box[2:], (0, 0, 255), 2)
            flag = 1
        #     filer.write('_'*20)
        # if flag == 1 and idx == rand_select and cate_probs[idx][ind] > 0.5:
        # if cate_probs[idx][ind] > 0.5:
        #     box = boxes[idx].numpy().astype(int)
        #     c_obj = c_objs[idx]
        #     attr_prob = attr_probs[idx]
        #     cate_prob = cate_probs[idx][ind]
        #     filer.write(f'\nobject confidence: {c_obj:.2f}\n')
        #     filer.write('category: \n')
        #     if ind in unseen_cate_id:
        #         seen = 'unseen'
        #     else:
        #         seen = 'seen'
        #     filer.write(f'\t{seen} {id2category[ind.item()]} {cate_prob:.2f}\n')
        #     filer.write('pos attribute: \n')
        #     att_values, att_inds = torch.sort(attr_prob, descending=True, dim=-1)
        #     for i_att in range(50):
        #         if att_inds[i_att] in unseen_att_id:
        #             seen = 'unseen'
        #         else:
        #             seen = 'seen'
        #         filer.write(f'\t{seen} {id2att[att_inds[i_att].item()]} {att_values[i_att]:.2f}\n')
        #     filer.write('neg attribute: \n')
        #     for i_att in list(range(len(attr_prob) - 3, len(attr_prob)))[::-1]:
        #         if att_inds[i_att] in unseen_att_id:
        #             seen = 'unseen'
        #         else:
        #             seen = 'seen'
        #         filer.write(f'\t{seen} {id2att[att_inds[i_att].item()]} {att_values[i_att]:.2f}\n')
        #
        #     img = cv2.rectangle(img, box[:2], box[2:], (0, 0, 255), 2)
        #     flag = 2
    cv2.imwrite(file_name.replace('.jpg', '.png'), img)
    # cv2.waitKey(0)


if __name__ == '__main__':
    json_data = json.load(open('/Users/kyanchen/Documents/COCO/annotations/instances_val2017.json'))
    img_folder = '/Users/kyanchen/Documents/COCO/val2017'
    id2name = {x['id']: x['name'] for x in json_data['categories']}
    id2images = {}
    id2instances = {}
    for data in json_data['images']:
        img_id = str(data['id'])
        data['file_name'] = f'{data["id"]:012d}.jpg'
        id2images[img_id] = data
    for data in json_data['annotations']:
        img_id = str(data['image_id'])
        data['name'] = id2name[data['category_id']]
        id2instances[img_id] = id2instances.get(img_id, []) + [data]

    for k, v in id2instances.items():
        img_id = k
        file_name = img_folder+f'/{int(img_id):012d}.jpg'
        img = cv2.imread(file_name)
        filer = open(file_name.replace('.jpg', '.txt').replace('val2017', 'cate_results'), 'w')
        instances = id2instances[img_id]
        for instance in instances:
            cate = id2name[instance['category_id']]
            color = PALETTE[CLASSES.index(cate)]
            x, y, w, h = instance['bbox']
            filer.write(f'{cate}\n')
            img = cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0,0,255), 2)
        filer.close()
        cv2.imwrite(file_name.replace('.jpg', '.png').replace('val2017', 'cate_results'), img)