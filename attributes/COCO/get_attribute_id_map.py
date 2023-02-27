import json
import pickle

# pkl_path = r"D:\Dataset\COCO\cocottributes_eccv_version.pkl"
# f = open(pkl_path, 'rb')
# data = pickle.load(f, encoding='latin1')
# pickle.dump(data, open(r'D:\Dataset\COCO\attributes_2014.pkl', 'wb'))

pkl_path = r"D:\Dataset\COCO\attributes_2014.pkl"
f = open(pkl_path, 'rb')
data = pickle.load(f)
attributes = data['attributes']
attributes = sorted(attributes, key=lambda x: x['id'])
attr_names = [item['name'] for item in attributes]
attr_names = [x.split('/') for x in attr_names]
attr_names = [[m.strip() for m in x] for x in attr_names]
attribute_map = {'num_attributes': 0, 'num_sub_attributes': 0, 'id2attribute': {}, 'attribute2id': {}}
attribute_map['id2attribute'] = {k: v for k, v in enumerate(attr_names)}
for k, v in enumerate(attr_names):
    for v_sub in v:
        attribute_map['attribute2id'][v_sub] = k
attribute_map['num_attributes'] = len(attribute_map['id2attribute'])
attribute_map['num_sub_attributes'] = len(attribute_map['attribute2id'])
json.dump(attribute_map, open(r'attribute_id_map.json', 'w'), indent=4)