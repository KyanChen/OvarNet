import json

import pandas as pd


data = pd.read_csv('results.csv', sep='|').iloc[:, -1]

# json.dump(list(data), open('captions_extracted.json', 'w'), indent=4)

json_data = {'num': len(data), 'captions': list(data)}
json.dump(json_data, open('../caption_all/Flickr30k_all_extracted.json', 'w'), indent=4)