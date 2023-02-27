import json

import tqdm
from textblob import TextBlob
from textblob import Word
from textblob.taggers import NLTKTagger
nltk_tagger = NLTKTagger()
import numpy as np

datas = json.load(open('infos/all_gather_atts.json', 'r'))['atts']
atts = {}
atts_keep = {}
for data in tqdm.tqdm(datas):
    key, value = data
    key = key.lower()
    if value < 5:
        continue
    if len(key.split(' ')) > 1:
        atts[key] = atts.get(key, 0) + value
        continue
    blob = TextBlob(key, pos_tagger=nltk_tagger)
    tags = [x.split('/')[1] for x in blob.parse().split(' ')]
    # print(blob.parse())
    tag = np.array([tag.split('PR')[0].split('RB')[0].split('WP')[0] for tag in tags])  # tag.split('NN')[0].split('PR')[0].split('VB')[0].split('RB')
    if np.any(tag=='IN') or np.any(tag=='VB') or np.any(tag=='VBZ') or np.any(tag=='') or np.any(tag=='LS') or np.any(tag=='UH') or np.any(tag=='DT') or np.any(tag=='TO') or np.any(tag=='CC') or np.any(tag=='CD') or np.any(tag=='MD') or np.any(tag=='FW') or np.any(tag=='EX'):  # 96415
        atts[key] = atts.get(key, 0) + value
    else:
        atts_keep[key] = atts_keep.get(key, 0) + value

atts = sorted(atts.items(), key=lambda kv: kv[1], reverse=True)
return_atts = {'num_atts': len(atts), 'atts': atts}
json.dump(return_atts, open(f'infos/all_gather_atts_need_tag.json', 'w'), indent=4)

atts_keep = sorted(atts_keep.items(), key=lambda kv: kv[1], reverse=True)
return_atts_keep = {'num_atts': len(atts_keep), 'atts': atts_keep}
json.dump(return_atts_keep, open(f'infos/all_gather_atts_keep_need_tag.json', 'w'), indent=4)
