import json
import os

import pandas
import tqdm
import multiprocessing


def get_key_freq(src_keys, target_data, path, pid):
    kv_dict = {}
    for key in tqdm.tqdm(src_keys):
        key = key.lower().strip('.').strip('?').strip('!').strip('\"')
        count_num = 0
        for tgt_text in target_data:
            if pandas.isna(tgt_text):
                continue
            try:
                tgt_text_list = tgt_text.lower().strip().split(' ')
                show_times = []
                for k in key.split(' '):
                    show_times.append(tgt_text_list.cout(k))
                count_num = min(show_times)

            except Exception as e:
                print(e)
        kv_dict[key] = count_num
    json.dump(kv_dict, open(path + f'/split_{pid}_with_freq.json', 'w'), indent=4)


def split_json_data(text_list, split_num, path):
    os.makedirs(path, exist_ok=True)
    n_item_per_slice = len(text_list) // split_num
    for i in range(split_num):
        start = i * n_item_per_slice
        end = min(start + n_item_per_slice, len(json_data))
        json.dump(text_list[start: end], open(path+f'/split_{i}.json', 'w'))


def gather_all(path, split_num):
    return_data = {'num_atts': 0, 'atts': {}}
    for i in range(split_num):
        data = json.load(open(path + f'/split_{i}_with_freq.json', 'r'))
        return_data['atts'].update(data)
    return_data['atts'] = sorted(return_data['atts'].items(), key=lambda kv: kv[1], reverse=True)
    return_data['num_atts'] = len(return_data['atts'])
    return return_data


if __name__ == '__main__':
    src_data = json.load(open('attributes.json', 'r', encoding='gbk'))
    src_keys = [m.split(',')[0] for x in src_data['attribute_tree'] for m in list(x.values())[0]]

    target_data = json.load(open('../../gather_infos/infos/all_gather_atts.json', 'r'))['atts']
    target_data_dict = {}
    for k, v in target_data:
        target_data_dict[k] = v
    src_data = {}
    for key in tqdm.tqdm(src_keys):
        src_data[key] = target_data_dict.get(key, 0)

    return_data = {'num_atts': 0, 'atts': {}}
    return_data['atts'] = sorted(src_data.items(), key=lambda kv: kv[1], reverse=True)
    return_data['num_atts'] = len(return_data['atts'])

    json.dump(return_data, open('VAW_attributes_with_freq.json', 'w'), indent=4)
