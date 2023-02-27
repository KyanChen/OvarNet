import json

from tqdm import tqdm

data_caption = json.load(open('infos/all_caption_extracted_categories.json', 'r'))['categories']
data_categories = json.load(open('infos/all_objects_with_freq_filtered.json', 'r'))['categories']
categories = {}

all_data = [data_categories, data_caption]
for idx, data_split in enumerate(all_data):
    for data in tqdm(data_split):
        key, value = data
        key = key.lower()
        # key, value = 'wooden', 200000
        if len(key) < 2:  # 425836
            continue

        categories[key] = categories.get(key, 0) + value
    if idx == 0:
        for k, v in categories.items():
            categories[k] = 10 * v

categories = sorted(categories.items(), key=lambda kv: kv[1], reverse=True)
return_categories = {'num_categories': len(categories), 'categories': categories}
json.dump(return_categories, open(f'infos/all_gather_categories.json', 'w'), indent=4)

