import json
import os
import glob

in_path = '../../attributes'
files = glob.glob(in_path+'/*/'+'attributes.json')
l2_attributes = set()
for file in files:
    attrs = json.load(open(file, 'r'))
    for att in attrs['attribute_tree']:
        for k, v in att.items():
            if v is None:
                l2_attributes.add(k)
            elif isinstance(v, list):
                for sub_att in v:
                    l2_attributes.add(sub_att)
l2_attributes = list(l2_attributes)
json_data = {'num_attributes': len(l2_attributes), 'l2_attributes': l2_attributes}
json.dump(json_data, open('../../attributes/all_attributes.json', 'w'), indent=4, ensure_ascii=False)
