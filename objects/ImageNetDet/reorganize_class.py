import json

import pandas as pd

data = open('classes.txt', 'r').readlines()
data = [x.strip() for x in data]
data = [x.lower() for x in list(data)]
json.dump(data, open('classes_extracted.json', 'w'), indent=4)
pass