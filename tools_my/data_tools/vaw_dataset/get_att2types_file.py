import json
import os

file = '/Users/kyanchen/Documents/VAW/attribute_types.json'
type_data = json.load(open(file, 'r'))
att2types = {'id2type': {}, 'att2typeid': {}}
types = list(type_data.keys())
for idx in range(len(types)):
    att2types['id2type'][idx] = types[idx]
for k, v in type_data.items():
    for att in v:
        att2types['att2typeid'][att] = types.index(k)
att2types['num_att'] = len(att2types['att2typeid'])
att2types['num_type'] = len(att2types['id2type'])
json.dump(att2types, open(os.path.dirname(file)+'/att2types.json', 'w'), indent=4)

