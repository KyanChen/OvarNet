import time

import tqdm
from googletrans import Translator
import json

# 设置Google翻译服务地址
dataset = 'VG'
pattern = 'attributes'

translator = Translator(service_urls=['translate.google.cn'])

# txt_path = f'../../attributes/{dataset}/{pattern}.txt'
# txt_data_lines = open(txt_path, 'r').readlines()
# att_data = {}


json_path = f'{pattern}.json'
json_data_lines = json.load(open(json_path, 'r'))
all_attributes = []
for data in tqdm.tqdm(json_data_lines):
    attributes = data['attributes']
    for att in attributes:
        all_attributes += att.get('attributes', [])

all_attributes = list(set(all_attributes))
json.dump(all_attributes, open(f'{pattern}_extracted.json', 'w'))

