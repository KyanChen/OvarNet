import json


data_400m = json.load(open('../captions/caption_all/laion400m_extracted_atts.json', 'r'))['atts']
data_2b = json.load(open('../captions/caption_all/laion2b_extracted_atts.json', 'r'))['atts']
atts = {}

all_data = [data_400m, data_2b]
for data_split in all_data:
    for data in data_split:
        key, value = data
        key = key.lower()
        if value < 5:  # 2: 900W; 5: 280W
            continue
        atts[key] = atts.get(key, 0) + value

atts = sorted(atts.items(), key=lambda kv: kv[1], reverse=True)
return_atts = {'num_atts': len(atts), 'atts': atts}
json.dump(return_atts, open(f'../captions/caption_all/laion_extracted_atts.json', 'w'), indent=4)
