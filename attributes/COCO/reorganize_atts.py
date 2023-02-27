import json

data = json.load(open(r"attributes.json", 'r', encoding='utf-8'))
data = data['l1_attributes']
json_data = {'num': len(data), 'attributes': list(data)}
json.dump(json_data, open('../attribute_all/COCO_all_extracted.json', 'w'), indent=4)