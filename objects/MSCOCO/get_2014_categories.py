import json

file = r"D:\Dataset\COCO\instances_train2014.json"
data = json.load(open(file, 'r'))
json.dump(data['categories'], open('categories.json', 'w'), indent=4)
pass