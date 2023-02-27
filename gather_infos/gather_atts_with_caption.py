import json

import tqdm
from textblob import TextBlob
from textblob import Word
from textblob.taggers import NLTKTagger
nltk_tagger = NLTKTagger()
import numpy as np

data_caption = json.load(open('infos/all_caption_extracted_atts.json', 'r'))['atts']
data_atts = json.load(open('infos/all_attributes_with_freq_filtered.json', 'r'))['atts']
atts = {}

all_data = [data_atts, data_caption]
for idx, data_split in enumerate(all_data):
    for data in tqdm.tqdm(data_split):
        key, value = data
        key = key.lower()
        # key, value = 'wooden', 200000
        if len(key) < 2:  # 425836
            continue
        # if idx == 0 and value < 5:  # 254368
        #     continue
        # if idx == 1 and value < 50:
        #     continue
        # blob = TextBlob(key, pos_tagger=nltk_tagger)
        # tags = [x.split('/')[1] for x in blob.parse().split(' ')]
        # # print(blob.parse())
        # tag = np.array([tag.split('PR')[0].split('RB') for tag in tags])  # tag.split('NN')[0].split('PR')[0].split('VB')[0].split('RB')
        # if np.all(tag=='IN') or np.all(tag=='') or np.all(tag=='DT') or np.all(tag=='TO') or np.all(tag=='CC') or np.all(tag=='CD') or np.all(tag=='MD') or np.all(tag=='FW'):  # 96415
        #     continue

        atts[key] = atts.get(key, 0) + value
    if idx == 0:
        for k, v in atts.items():
            atts[k] = 10 * v

atts = sorted(atts.items(), key=lambda kv: kv[1], reverse=True)
return_atts = {'num_atts': len(atts), 'atts': atts}
json.dump(return_atts, open(f'infos/all_gather_atts.json', 'w'), indent=4)

