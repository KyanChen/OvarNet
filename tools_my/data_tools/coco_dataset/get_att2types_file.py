import json
import os

file = '/Users/kyanchen/My_Code/prompt/objects/MSCOCO/categories.json'
save_path = '../../../attributes/COCO'

type_data = json.load(open(file, 'r'))
att2types = {'id2type': {}, 'att2typeid': {}}
types = list(set([x["supercategory"] for x in type_data]))
for idx in range(len(types)):
    att2types['id2type'][idx] = types[idx]
for item in type_data:
    att2types['att2typeid'][item['name']] = types.index(item['supercategory'])
att2types['num_category'] = len(att2types['att2typeid'])
att2types['num_type'] = len(att2types['id2type'])
json.dump(att2types, open(save_path+'/category2types.json', 'w'), indent=4)

