import json
from collections import OrderedDict

import tqdm

data_root = '../../../attributes/COCO'
data_path = '/Users/kyanchen/Documents/COCO/annotations/instances_train2017.json'


json_file = 'instances_train2017'
# json_file = 'lvis_v1_train' if pattern == 'train' else 'instances_val2017'
json_data = json.load(open(data_path, 'r'))
id2name = {x['id']: x['name'] for x in json_data['categories']}
instances = json_data['annotations']

freq_attr = OrderedDict()
for key, value in id2name.items():
    freq_attr[value] = {'total': 0}
for instance in tqdm.tqdm(instances):
    freq_attr[id2name[instance['category_id']]]['total'] = freq_attr[id2name[instance['category_id']]]['total'] + 1

json.dump(freq_attr, open(data_root + '/category_freq_wo_sort.json', 'w'), indent=4)
freq_attr = sorted(freq_attr.items(), key=lambda kv: kv[1]['total'], reverse=True)
json.dump(freq_attr, open(data_root + '/category_freq_w_sort.json', 'w'), indent=4)

