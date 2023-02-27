import glob
import os
import json

files = glob.glob('attribute_all/*_extracted.json')
json_data = {'num': 0, 'attributes': []}
for file in files:
    data = json.load(open(file, 'r'))
    json_data['attributes'] += data['attributes']
data = json_data['attributes']

data_tmp = []
for x in data:
    if ('/' in x) and ('w/' not in x):
        x = x.split('/')
        data_tmp += x
    else:
        data_tmp += [x]
data = data_tmp
data = [x.strip().lower() for x in data]
data = [x for x in data if x != '']
json_data['attributes'] = list(set(data))
json_data['num'] = len(json_data['attributes'])
json.dump(json_data, open('attribute_all/all_attributes.json', 'w'), indent=4)