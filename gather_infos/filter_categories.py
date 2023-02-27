import json


datas = json.load(open('infos/all_objects_with_freq.json', 'r'))['objects']
categories = {}

for data in datas:
    key, value = data
    key = key.lower()
    # if value < 50:  # 2: 900W; 5: 280W
    #     continue
    if len(key) < 2:
        continue
    categories[key] = categories.get(key, 0) + value

categories = sorted(categories.items(), key=lambda kv: kv[1], reverse=True)
return_categories = {'num_categories': len(categories), 'categories': categories}
json.dump(return_categories, open(f'infos/all_objects_with_freq_filtered.json', 'w'), indent=4)
