import time

from googletrans import Translator
import json

# 设置Google翻译服务地址
translator = Translator(service_urls=['translate.google.cn'])

all_att = {'att': ['color', 'texture', 'shape', 'pattern'], 'levels': 2}
color = {'color': ['black', 'brown', 'gray', 'green', 'orange', 'red', 'white', 'yellow']}
texture = {'texture': ['furry', 'metallic', 'rough', 'shiny', 'smooth', 'wet', 'wooden']}
shape = {'shape': ['long', 'rectangular', 'round']}
pattern = {'pattern': ['spotted', 'striped']}

json_data = {}
for k, v in all_att.items():
    if k == 'att':
        json_data['l1_attributes'] = v
    else:
        json_data[k] = v

json_data['attribute_tree'] = []
for att in json_data['l1_attributes']:
    sub_atts = []
    for sub_att in eval(att)[att]:
        translation = translator.translate(sub_att, src='en', dest='zh-CN')
        sub_atts.append(sub_att + ',' + translation.text)
        time.sleep(0.1)
    translation = translator.translate(att, src='en', dest='zh-CN')
    json_data['attribute_tree'].append({att + ',' + translation.text: sub_atts})
    time.sleep(0.1)

json.dump(json_data, open('/Users/kyanchen/Code/CLIP_Prompt/attributes/Attribute_learning/attributes.json', 'w'), indent=4, ensure_ascii=False)

