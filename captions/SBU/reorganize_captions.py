import json

import pandas as pd

data = json.load(open(r"D:\Dataset\SBU\sbu-captions-all.json", 'r'))
data = data['captions']
json_data = {'num': len(data), 'captions': list(data)}
json.dump(json_data, open('../caption_all/SBU_all_extracted.json', 'w'), indent=4)