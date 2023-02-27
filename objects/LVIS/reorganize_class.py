import json

import pandas as pd

data = json.load(open('categories.json', 'r'))
data = list(set([x['name'].replace('_', ' ') for x in data]))
json.dump(data, open('classes_extracted.json', 'w'), indent=4)
pass