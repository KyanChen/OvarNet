import json
import pandas as pd

data = json.load(open(r"D:\Dataset\VG\region_descriptions.json", 'r'))
data = [x['regions'] for x in data]
data = [m['phrase'] for x in data for m in x]
json_data = {'num': len(data), 'captions': list(data)}
json.dump(json_data, open('../caption_all/VG_all_extracted.json', 'w'), indent=4)