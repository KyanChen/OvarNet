import json
json.dump(json.load(open('classes_extracted.json', 'r')), open('classes_extracted.json', 'w'), indent=4)