import json


datas = json.load(open('infos/all_attributes_with_freq.json', 'r'))['atts']
atts = {}

for data in datas:
    key, value = data
    key = key.lower()
    # if value < 5:  # 2: 900W; 5: 280W
    #     continue
    if len(key) < 3:
        continue
    atts[key] = atts.get(key, 0) + value

atts = sorted(atts.items(), key=lambda kv: kv[1], reverse=True)
return_atts = {'num_atts': len(atts), 'atts': atts}
json.dump(return_atts, open(f'infos/all_attributes_with_freq_filtered.json', 'w'), indent=4)
