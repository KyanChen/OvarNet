import json

file = '../../../attributes/OVAD/ovad2000.json'
save_path = '../../../attributes/OVAD'

type_data = json.load(open(file, 'r'))['attributes']
type2atts = {}
for data in type_data:
    type2atts[data['parent_type']] = type2atts.get(data['parent_type'], set({}))
    type2atts[data['parent_type']].add(data['type'])
for k, v in type2atts.items():
    type2atts[k] = list(v)
json.dump(type2atts, open(save_path+'/attribute_parent_types.json', 'w'), indent=4)