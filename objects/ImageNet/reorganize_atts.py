import json

data = json.load(open(r"classes.json", 'r', encoding='utf-8'))
data = data['l1_attributes']
data = [x.split('/') for x in data]
data = [m.strip() for x in data for m in x]
json_data = {'num': len(data), 'attributes': list(data)}
json.dump(json_data, open('../object_all/ImageNetAtt_all_extracted.json', 'w'), indent=4)