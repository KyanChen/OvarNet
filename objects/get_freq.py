import json
import os

import pandas
import tqdm
import multiprocessing


def get_key_freq(src_keys, target_data, path, pid):
    kv_dict = {}
    for key in tqdm.tqdm(src_keys):
        key = key.lower().strip('.').strip('?').strip('!').strip('\"').strip('`').strip('@').strip('\'').strip()
        if len(key) < 3:
            continue
        count_num = 0
        for tgt_text in target_data:
            if pandas.isna(tgt_text):
                continue
            try:
                tgt_text_list = tgt_text.lower().strip('.').strip('?').strip('!').strip('\"').strip('`').strip('@').strip('\'').strip().split(' ')
                show_times = []
                for k in [x.strip() for x in key.split(' ')]:
                    show_times.append(tgt_text_list.count(k))
                count_num += min(show_times)
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
    return_data = {'num_objects': 0, 'objects': {}}
    for i in range(split_num):
        data = json.load(open(path + f'/split_{i}_with_freq.json', 'r'))
        for k, v in data.items():
            return_data['objects'][k] = return_data['objects'].get(k, 0) + v
    return_data['objects'] = sorted(return_data['objects'].items(), key=lambda kv: kv[1], reverse=True)
    return_data['num_objects'] = len(return_data['objects'])
    return return_data


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    n_process = 64

    src_data = json.load(open('../gather_infos/infos/all_objects.json', 'r'))['objects']
    data_slice_list = []
    n_item_per_slice = len(src_data) // n_process
    for i in range(n_process):
        start = i * n_item_per_slice
        end = start + n_item_per_slice
        if i == n_process - 1:
            end = len(src_data)
        data_slice_list.append(src_data[start: end])

    target_data = json.load(open('../captions/caption_all/caption_seg_word.json', 'r'))['captions']

    tmp_path = 'object_all/tmp'
    os.makedirs(tmp_path, exist_ok=True)
    process_list = []
    for pid in range(n_process):
        print('pid {}'.format(pid))
        process_list.append(
            multiprocessing.Process(target=get_key_freq, args=(data_slice_list[pid], target_data, tmp_path, pid))
        )
    [p.start() for p in process_list]
    [p.join() for p in process_list]

    json_data = gather_all(tmp_path, split_num=n_process)
    json.dump(json_data, open('../gather_infos/infos/all_objects_with_freq.json', 'w'), indent=4)
