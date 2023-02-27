import json


data_laion = json.load(open('caption_all/laion_extracted_atts.json', 'r'))['atts']
data_other = json.load(open('../gather_infos/infos/other_extracted_atts.json', 'r'))['atts']
atts = {}

all_data = [data_laion, data_other]
for data_split in all_data:
    for data in data_split:
        key, value = data
        key = key.lower()
        if value < 50:  # 2: 900W; 5: 280W
            continue
        atts[key] = atts.get(key, 0) + value

atts = sorted(atts.items(), key=lambda kv: kv[1], reverse=True)
return_atts = {'num_atts': len(atts), 'atts': atts}
json.dump(return_atts, open(f'caption_all/all_caption_extracted_atts.json', 'w'), indent=4)
