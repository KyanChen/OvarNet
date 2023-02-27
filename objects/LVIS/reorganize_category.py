import json

data = json.load(open(r"classes_extracted.json", 'r', encoding='utf-8'))
data = data
json_data = {'num': len(data), 'attributes': list(data)}
json.dump(json_data, open('../object_all/LVIS_all_extracted.json', 'w'), indent=4)