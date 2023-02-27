import json
from collections import OrderedDict

import tqdm

data_root = '/expand_data/datasets/LSA/'
attr_dict = json.load(open('../../../attributes/LSA/common2rare.json'))
all_attr_dict = {}
all_attr_dict.update(attr_dict['common'])
all_attr_dict.update(attr_dict['rare'])

json_data = json.load(open(data_root + 'train.json'))
freq_attr = OrderedDict()
for key, value in all_attr_dict.items():
    freq_attr[key] = {'total': 0}

for img_item in tqdm.tqdm(json_data):
    img_id = img_item['image_id']
    objects = img_item['objects']
    for i_object in objects:
        for item in i_object['attributes']:
            try:
                freq_attr[item]['total'] = freq_attr[item]['total'] + 1
            except Exception as e:
                print(e)


json.dump(freq_attr, open(data_root + 'attr_freq_wo_sort.json', 'w'), indent=4)
freq_attr = sorted(freq_attr.items(), key=lambda kv: kv[1]['total'], reverse=True)
json.dump(freq_attr, open(data_root + 'attr_freq_w_sort.json', 'w'), indent=4)

