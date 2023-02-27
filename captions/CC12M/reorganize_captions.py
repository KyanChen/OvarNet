import json

import pandas as pd

data = pd.read_csv('cc12m.tsv', header=None, delimiter='\t')

data = list(data.iloc[:, 1])
json_data = {'num': len(data), 'captions': list(data)}
json.dump(json_data, open('../caption_all/CC12M_all_extracted.json', 'w'), indent=4)