import json

import pandas as pd

data = json.load(open(r"D:\Dataset\VizWiz-Captions\annotations\val.json", 'r'))
data = data['annotations']
data = [x['caption'] for x in data]

json_data = {'num': len(data), 'captions': list(data)}
json.dump(json_data, open('../caption_all/VizWiz_val_extracted.json', 'w'), indent=4)