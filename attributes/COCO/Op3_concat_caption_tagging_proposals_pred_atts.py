import json
import mmcv
import numpy as np
from tqdm import tqdm
import sys
sys.path.append(sys.path[0]+'/../..')
# from mmdet.core import bbox_overlaps
from tools_my.cache_data_tools.redis_utils import RedisHelper

parent_folder = '../../data/COCO/annotations'
# parent_folder = '/Users/kyanchen/Documents/COCO/annotations'
json_file = parent_folder+'/train_2017_caption_tagging_with_proposals.json'
ori_data = json.load(open(json_file, 'r'))

pred_atts = '../../tools/train2017_proposals_predatts.pkl'
pred_atts = mmcv.load(pred_atts)

redis_helper = RedisHelper()
if not redis_helper.redis:
    redis_helper.init_redis()

flag_id_start = 0
prefix = 'cky_'
for img_id, data in tqdm(ori_data.items()):
    proposals = np.array(data['proposals'])
    flag_id_end = flag_id_start + len(proposals)
    proposals_atts = pred_atts[flag_id_start: flag_id_end]
    proposals = np.concatenate((proposals, proposals_atts), axis=-1)  # 6 [xywh,conf,class]+ 606
    data['proposals'] = proposals.tolist()
    data['img_id'] = img_id
    img_id = prefix + str(img_id)
    redis_helper.redis.set(img_id, json.dumps(data))
    flag_id_start = flag_id_end


