import glob
import json

import tqdm

CC_folder = '../../data/CC3M/CC3MImages'
all_img = glob.glob(CC_folder+'/*/*.jpg')
valid_img_keys = []
for img in tqdm.tqdm(all_img):
    try:
        json_data = json.load(open(img.replace('.jpg', '.json')))
        if json_data['status'] == 'success':
            valid_img_keys.append(json_data['key'])
        else:
            print(json_data['status'])
    except Exception as e:
        print(e)
print('all data: ', len(all_img))
print('valid data: ', len(valid_img_keys), '  ', len(set(valid_img_keys)))
json.dump(valid_img_keys, open('train_CC_keys.json', 'w'), indent=4)
