import json

import tqdm

data_400m = json.load(open('../captions/caption_all/laion400m_extracted_categories.json', 'r'))['categories']
data_2b = json.load(open('../captions/caption_all/laion2b_extracted_categories.json', 'r'))['categories']
categories = {}

all_data = [data_400m, data_2b]
for data_split in all_data:
    for data in tqdm.tqdm(data_split):
        key, value = data
        key = key.lower()
        if value < 5:
            continue
        categories[key] = categories.get(key, 0) + value

categories = sorted(categories.items(), key=lambda kv: kv[1], reverse=True)
return_categories = {'num_categories': len(categories), 'categories': categories}
json.dump(return_categories, open(f'../captions/caption_all/laion_extracted_categories.json', 'w'), indent=4)
