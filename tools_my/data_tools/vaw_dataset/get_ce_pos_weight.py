import json
from collections import OrderedDict

import tqdm

data_root = '/data1/kyanchen/prompt/data/VAW/'
attr_dict = json.load(open(data_root + 'attribute_index.json'))


json_file_list = ["train_part1.json", "train_part2.json"]
json_data = [json.load(open(data_root + x)) for x in json_file_list]
instances = []
[instances.extend(x) for x in json_data]
freq_attr = OrderedDict()
for key, value in attr_dict.items():
    freq_attr[key] = {'pos': 0, 'neg': 0, 'total': 0}
for instance in tqdm.tqdm(instances):
    positive_attributes = instance['positive_attributes']
    negative_attributes = instance['negative_attributes']
    for item in positive_attributes:
        freq_attr[item]['pos'] = freq_attr[item]['pos'] + 1
        freq_attr[item]['total'] = freq_attr[item]['total'] + 1

    for item in negative_attributes:
        freq_attr[item]['neg'] = freq_attr[item]['neg'] + 1
        freq_attr[item]['total'] = freq_attr[item]['total'] + 1

json.dump(freq_attr, open(data_root + 'attr_freq_wo_sort.json', 'w'), indent=4)
freq_attr = sorted(freq_attr.items(), key=lambda kv: kv[1]['total'], reverse=True)
json.dump(freq_attr, open(data_root + 'attr_freq_w_sort.json', 'w'), indent=4)

