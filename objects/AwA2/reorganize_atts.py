import json

data = json.load(open(r"classes.json", 'r', encoding='utf-8'))
data = data['l1_attributes']
json_data = {'num': len(data), 'attributes': list(data)}
json.dump(json_data, open('../object_all/AwA2_all_extracted.json', 'w'), indent=4)