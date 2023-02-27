import json

data = json.load(open(r"D:\Dataset\VG\unzip_data\attributes.json", 'r', encoding='utf-8'))
data = [m.get('attributes', []) for x in data for m in x['attributes']]
data = [m for x in data for m in x]
data = list(set(data))
json_data = {'num': len(data), 'attributes': list(data)}
json.dump(json_data, open('../attribute_all/VG_all_extracted.json', 'w'), indent=4)