import json

import tqdm
from textblob import TextBlob
from textblob import Word
from textblob.taggers import NLTKTagger
nltk_tagger = NLTKTagger()
import numpy as np

datas = json.load(open('infos/all_gather_atts.json', 'r'))['atts']
atts = {}

for data in tqdm.tqdm(datas):
    key, value = data
    key = key.lower()
    if value < 5:
        continue
    blob = TextBlob(key, pos_tagger=nltk_tagger)
    tags = [x.split('/')[1] for x in blob.parse().split(' ')]
    # print(blob.parse())
    tag = np.array([tag.split('PR')[0].split('RB')[0].split('WP')[0] for tag in tags])  # tag.split('NN')[0].split('PR')[0].split('VB')[0].split('RB')
    if np.all(tag=='IN') or np.all(tag=='VB') or np.all(tag=='VBZ') or np.all(tag=='') or np.any(tag=='LS') or np.all(tag=='UH') or np.all(tag=='DT') or np.all(tag=='TO') or np.all(tag=='CC') or np.all(tag=='CD') or np.all(tag=='MD') or np.all(tag=='FW') or np.all(tag=='EX'):  # 96415
        continue

    atts[key] = atts.get(key, 0) + value

atts = sorted(atts.items(), key=lambda kv: kv[1], reverse=True)
return_atts = {'num_atts': len(atts), 'atts': atts}
json.dump(return_atts, open(f'infos/all_gather_atts_filtered.json', 'w'), indent=4)
