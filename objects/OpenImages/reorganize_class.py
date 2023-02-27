import json

import pandas as pd

data = pd.read_csv('class_box_600.csv').iloc[:, 1]
data = [x.lower() for x in list(data)]
json.dump(data, open('classes_extracted.json', 'w'), indent=4)
pass