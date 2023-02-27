import glob
import os
import json

files = glob.glob('caption_all/*_extracted.json')
json_data = {'num': 0, 'captions': []}
for file in files:
    data = json.load(open(file, 'r'))
    json_data['captions'] += data['captions']
json_data['num'] = len(json_data['captions'])
json.dump(json_data, open('caption_all/caption_seg_word.json', 'w'), indent=4)