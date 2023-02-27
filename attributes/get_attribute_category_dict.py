import json
import random
from collections import OrderedDict

import tqdm

data_root = '/data/kyanchen/prompt/data/VAW/'
attr_dict = json.load(open(data_root + 'attribute_index.json'))

json_file_list = ["train_part1.json", "train_part2.json", 'test.json', 'val.json']
json_data = [json.load(open(data_root + x)) for x in json_file_list]
instances = []
[instances.extend(x) for x in json_data]
freq_attr = OrderedDict()

for instance in tqdm.tqdm(instances):
    positive_attributes = instance['positive_attributes']
    negative_attributes = instance['negative_attributes']
    category = instance['object_name']
    freq_attr[category] = freq_attr.get(category, {})
    freq_attr[category]['n_instance'] = freq_attr[category].get('n_instance', 0) + 1
    freq_attr[category]['pos'] = freq_attr[category].get('pos', {})
    freq_attr[category]['neg'] = freq_attr[category].get('neg', {})
    for item in positive_attributes:
        freq_attr[category]['pos'][item] = freq_attr[category]['pos'].get(item, 0) + 1

    for item in negative_attributes:
        freq_attr[category]['neg'][item] = freq_attr[category]['neg'].get(item, 0) + 1

category_instances = OrderedDict()
for k, v in freq_attr.items():
    category_instances[k] = v['n_instance']
category_instances = sorted(category_instances.items(), key=lambda kv: kv[1], reverse=True)
json.dump(category_instances, open(data_root + 'category_instances.json', 'w'), indent=4)
json.dump(freq_attr, open(data_root + 'category_attr_pair.json', 'w'), indent=4)

# 2260个类别
n_category = len(category_instances)
total_test_instances = 0
test_category_instances = []
test_ids = []
while total_test_instances < 40000:
    random_id = random.randint(0, n_category)
    if random_id in test_ids:
        continue
    test_ids += [random_id]
    test_category_instance = category_instances[random_id]
    test_category_instances.append(test_category_instance)
    total_test_instances += test_category_instance[1]
# 41587
print(total_test_instances)

train_category_instances = []
for i in range(n_category):
    if i not in test_ids:
        train_category_instances.append(category_instances[i])
category_instances = {'train_category': train_category_instances, 'test_category': test_category_instances}
json.dump(category_instances, open(data_root + 'category_instances_split.json', 'w'), indent=4)




