import json

import pandas as pd

data = json.load(open("D:\Dataset\TextCaps\TextCaps_0.1_val.json", 'r'))
data = data['data']
data = [t for x in data for t in x['reference_strs']]

json_data = {'num': len(data), 'captions': list(data)}
json.dump(json_data, open('../caption_all/TextCaps_val_extracted.json', 'w'), indent=4)