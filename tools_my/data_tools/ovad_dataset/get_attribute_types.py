import json

file = '../../../attributes/OVAD/ovad2000.json'
save_path = '../../../attributes/OVAD'

type_data = json.load(open(file, 'r'))['attributes']
type2atts = {}
for data in type_data:
    type2atts[data['type']] = type2atts.get(data['type'], []) + [data['name']]
json.dump(type2atts, open(save_path+'/attribute_types.json', 'w'), indent=4)