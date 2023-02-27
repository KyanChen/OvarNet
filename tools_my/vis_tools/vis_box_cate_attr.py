import json
import pickle
import random
import shutil

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms

file_attr = '../../attributes/VAW/base2novel_att2id.json'
file_cate = '../../attributes/COCO/common2common_category2id_48_17.json'

att2id = {}
att_seen_unseen = {}
att2id_data = json.load(open(file_attr, 'r'))
att2id.update(att2id_data['base'])
att2id.update(att2id_data['novel'])
att_seen_unseen['seen'] = list(att2id_data['base'].keys())
att_seen_unseen['unseen'] = list(att2id_data['novel'].keys())

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

def vis(img, preds, file_name):
    # preds = torch.from_numpy(preds)
    boxes = preds[:, :4]
    c_objs = preds[:, 4]
    attr_probs = preds[:, 6:len(att2id)+6].sigmoid()
    cate_probs = preds[:, -len(category2id):].softmax(dim=-1)

    unseen_cate_id = [category2id[k] for k in category_seen_unseen['unseen']]
    unseen_att_id = [att2id[k] for k in att_seen_unseen['unseen']]

    cate_probs_fuse = torch.sqrt(cate_probs * c_objs[:, None])
    # cate_probs_fuse = cate_probs
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
    # scores = scores

    file_name = file_name.replace('val2017', 'result_att_cate_umbrella').replace('VAW', 'result_att_cate_umbrella')
    # values, inds = torch.max(cate_probs, dim=-1)
    filer = open(file_name.replace('.jpg', '.txt'), 'w')
    flag = 0
    rand_select = random.randint(0, 10)
    # rand_select = inds[0]
    for idx, ind in enumerate(pred_label):
        if ind in unseen_cate_id and flag == 0 and cate_probs[idx][ind] > 0.5:
        # if flag == 0 and cate_probs[idx][ind] > 0.9:
            if id2category[ind.item()] != 'skateboard':
                continue
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
            for i_att in range(60):
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
    if flag == 1:
        cv2.imwrite(file_name.replace('.jpg', '.png'), img)
    # cv2.waitKey(0)


if __name__ == '__main__':
    # att_preds = pickle.load(open('/Users/kyanchen/Documents/COCO/pred_box_free.pkl', 'rb'))
    att_preds = torch.load('/Users/kyanchen/Documents/COCO/pred_box_free.pth')
    # keys = [x for x in att_preds.keys() if 'vaw' in x]
    # for key in keys:
    #     img_id = key.split('_')[-1]
    #     try:
    #         shutil.copy(f'/Users/kyanchen/Documents/COCO/VG_100K/{img_id}.jpg',
    #                     f'/Users/kyanchen/Documents/COCO/VAW/{img_id}.jpg')
    #     except:
    #         shutil.copy(f'/Users/kyanchen/Documents/COCO/VG_100K_2/{img_id}.jpg',
    #                     f'/Users/kyanchen/Documents/COCO/VAW/{img_id}.jpg')
    coco_img_folder = '/Users/kyanchen/Documents/COCO/val2017'
    vaw_img_folder = '/Users/kyanchen/Documents/COCO/VAW'
    for k, v in att_preds.items():
        data_set, img_id = k.split('_')
        if data_set == 'coco':
            file_name = coco_img_folder + f'/{int(img_id):012d}.jpg'
        elif data_set == 'vaw':
            file_name = vaw_img_folder + f'/{img_id}.jpg'
        else:
            raise NameError
        img = cv2.imread(file_name)
        if img is None:
            print(file_name)
        else:
            vis(img, v, file_name=file_name)
