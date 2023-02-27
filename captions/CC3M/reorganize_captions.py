import json

import pandas as pd

data = pd.read_csv("D:\Dataset\CC3M\Validation_GCC-1.1.0-Validation.tsv", header=None, delimiter='\t')

data = data.iloc[:, 0]
json_data = {'num': len(data), 'captions': list(data)}
json.dump(json_data, open('../caption_all/CC3M_val_extracted.json', 'w'), indent=4)