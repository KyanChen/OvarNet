import glob
import os
import json

files = glob.glob('object_all/*_extracted.json')
json_data = {'num': 0, 'objects': []}
for file in files:
    data = json.load(open(file, 'r'))
    json_data['objects'] += data['attributes']
data = json_data['objects']

data_tmp = []
for x in data:
    if '(' in x:
        x = x.split('(')[0]
        data_tmp += [x]
    elif '\"' in x:
        x = x.replace('\"', '')
        data_tmp += [x]
    elif '\\' in x:
        x = x.replace('\\', '')
        data_tmp += [x]
    else:
        data_tmp += [x]
data = data_tmp
data = [x.strip().lower() for x in data]
json_data['objects'] = list(set(data))
json_data['num'] = len(json_data['objects'])
json.dump(json_data, open('object_all/all_objects.json', 'w'), indent=4)