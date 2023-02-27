import glob
import json
import math
import multiprocessing
import os
import numpy as np
import tqdm

def save_bp(pid, files):
    for file in tqdm.tqdm(files):
        data_json = json.load(open(file, 'r'))
        v = np.array(data_json['proposals'])
        v_area = v[:, 2] * v[:, 3]
        ind = np.argmax(v_area)
        biggest_box = v[ind][:4].tolist()
        data_json['biggest_proposal'] = biggest_box
        json.dump(data_json, open(file.replace('proposals_with_probs', 'propsoals_labels'), 'w'), indent=4)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    n_process = 64

    json_folder = '/data/kyanchen/prompt/data/CC3M/proposals_with_probs'
    files = glob.glob(json_folder + '/*.json')
    print(len(files))
    # 001535015
    split_keys = []
    n_item_per_slice = math.ceil(len(files) / n_process)
    for i in range(n_process):
        start = i * n_item_per_slice
        end = min(start + n_item_per_slice, len(files))
        split_keys.append(files[start:end])

    process_list = []
    for pid in range(n_process):
        print('pid {}'.format(pid))
        process_list.append(
            multiprocessing.Process(target=save_bp, args=(pid, split_keys[pid]))
        )
    [p.start() for p in process_list]
    [p.join() for p in process_list]
