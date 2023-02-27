import json

file = '../../../attributes/OVAD/ovad2000.json'
save_path = '../../../attributes/OVAD'

type_data = json.load(open(file, 'r'))['attributes']
head2atts = {}
for data in type_data:
    head2atts[data['freq_set']] = head2atts.get(data['freq_set'], []) + [data['name']]
json.dump(head2atts, open(save_path+'/head_tail.json', 'w'), indent=4)