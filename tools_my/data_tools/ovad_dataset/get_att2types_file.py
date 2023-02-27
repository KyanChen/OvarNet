import json
import os

file = '../../../attributes/OVAD/ovad2000.json'
save_path = '../../../attributes/OVAD'

type_data = json.load(open(file, 'r'))['attributes']
att2types = {'id2type': {}, 'att2typeid': {}}
types = list(set([x["type"] for x in type_data]))
for idx in range(len(types)):
    att2types['id2type'][idx] = types[idx]
for item in type_data:
    names = item["name"]
    att2types['att2typeid'][names] = types.index(item['type'])
att2types['num_category'] = len(att2types['att2typeid'])
att2types['num_type'] = len(att2types['id2type'])
json.dump(att2types, open(save_path+'/att2types.json', 'w'), indent=4)

