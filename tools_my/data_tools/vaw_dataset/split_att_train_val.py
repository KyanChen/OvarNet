import json
import os.path

freq_file = '/Users/kyanchen/Documents/VAW/attr_freq_w_sort.json'
freq_data = json.load(open(freq_file, 'r'))
common2rare = {'common': {}, 'rare': {}}
threshold = 100
for item in freq_data:
    freq = item[1]['total']
    if freq > threshold:
        common2rare['common'][item[0]] = freq
    else:
        common2rare['rare'][item[0]] = freq
common2rare['common_len'] = len(common2rare['common'])
common2rare['rare_len'] = len(common2rare['rare'])
json.dump(common2rare, open(os.path.dirname(freq_file)+'/common2rare_freq.json', 'w'), indent=4)
common2rare_att2id = {'common': {}, 'rare': {}}

idx = 0
for k, v in common2rare['common'].items():
    common2rare_att2id['common'][k] = idx
    idx += 1
for k, v in common2rare['rare'].items():
    common2rare_att2id['rare'][k] = idx
    idx += 1
json.dump(common2rare_att2id, open(os.path.dirname(freq_file)+'/common2rare_att2id.json', 'w'), indent=4)


common2common = {'common1': {}, 'common2': {}}
common2_tmp = []
for item in freq_data:
    freq = item[1]['total']
    if freq > 5000:
        common2common['common1'][item[0]] = freq
    elif freq <= 100:
        pass
    else:
        common2_tmp.append({item[0]: freq})

import random
random.shuffle(common2_tmp)
random.shuffle(common2_tmp)
for idx, item in enumerate(common2_tmp):
    if idx <= 0.1*(len(common2_tmp)):
        common2common['common2'].update(item)
    else:
        common2common['common1'].update(item)
common2common['common1_len'] = len(common2common['common1'])
common2common['common2_len'] = len(common2common['common2'])
json.dump(common2common, open(os.path.dirname(freq_file)+'/common2common_freq.json', 'w'), indent=4)
common2common_att2id = {'common1': {}, 'common2': {}}

idx = 0
for k, v in common2common['common1'].items():
    common2common['common1'][k] = idx
    idx += 1
for k, v in common2common['common2'].items():
    common2common['common2'][k] = idx
    idx += 1
json.dump(common2common, open(os.path.dirname(freq_file)+'/common2common_att2id.json', 'w'), indent=4)
