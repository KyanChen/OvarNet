import json

import tqdm

data_laion = json.load(open('caption_all/laion_extracted_categories.json', 'r'))['categories']
data_other = json.load(open('../gather_infos/infos/other_extracted_categories.json', 'r'))['categories']
categories = {}

all_data = [data_laion, data_other]
for data_split in all_data:
    for data in tqdm.tqdm(data_split):
        key, value = data
        key = key.lower()
        if value < 50:
            continue
        categories[key] = categories.get(key, 0) + value

categories = sorted(categories.items(), key=lambda kv: kv[1], reverse=True)
return_categories = {'num_categories': len(categories), 'categories': categories}
json.dump(return_categories, open(f'caption_all/all_caption_extracted_categories.json', 'w'), indent=4)
