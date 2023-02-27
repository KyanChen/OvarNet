import multiprocessing
import sys
import time
from googletrans import Translator
import json
import tqdm


def get_trans(att_data, q, levels, pid_id):
    translator = Translator(service_urls=['translate.google.cn'])
    # if pid_id == 0:
    tbar = tqdm.tqdm(total=len(att_data))
    for att_k, att_v in att_data.items():
        # if pid_id == 0:
        tbar.update(1)

        if levels == 2:
            sub_atts = []
            for sub_att in att_v:
                try_times = 0
                while True:
                    try:
                        translation = translator.translate(sub_att, src='en', dest='zh-CN')
                        break
                    except:
                        try_times += 1
                        if try_times > 20:
                            time.sleep(1)
                        time.sleep(0.1)
                sub_atts.append(sub_att + ',' + translation.text)
        else:
            sub_atts = None
        try_times = 0
        while True:
            try:
                translation = translator.translate(att_k, src='en', dest='zh-CN')
                break
            except:
                try_times += 1
                if try_times > 20:
                    time.sleep(1)
                time.sleep(0.2)
        q.append({att_k + ',' + translation.text: sub_atts})


def dic_slice(dic, start, end):
    _dic = dic
    keys = list(_dic.keys())
    _dic_slice = {}
    for key in keys[start: end]:   # 通过index方法，让列表自己找到索引值并返回
        _dic_slice[key] = _dic[key]
    return _dic_slice


if __name__ == '__main__':
    # 设置Google翻译服务地址
    dataset = 'VG'
    pattern = 'attributes'

    # txt_path = f'../../attributes/{dataset}/{pattern}.txt'
    # txt_data_lines = open(txt_path, 'r').readlines()
    # att_data = {}

    json_path = f'../../attributes/{dataset}/{pattern}_extracted.json'
    json_data_lines = json.load(open(json_path, 'r'))

    att_data = {}
    for att in json_data_lines:
        att_data[att] = att

    # for data in txt_data_lines:
    #     att, sub_att = data.split(':')
    #
    #     att = att.strip().split('_')[-1].replace('_', ' ').lower()
    #     sub_att = sub_att.strip().replace('_', ' ').lower().split(',')
    #     sub_att = [x.strip() for x in sub_att]
    #     att_data[att] = att_data.get(att, [])
    #     att_data[att] += sub_att

    # txt_data_lines = txt_data_lines[0].strip().split(' ')
    # for data in txt_data_lines:
    #     # att, sub_att = data.split('::')
    #     sub_att = data.split(":")[-1].replace(',', '/')
    #     att = sub_att
    #
    #     att = att.strip().replace('_', ' ').lower()
    #     sub_att = sub_att.strip().replace('_', ' ').lower().split('(')[0].strip()
    #     att_data[att] = att_data.get(att, [])
    #     att_data[att] += [sub_att]

    # for data in txt_data_lines:
    #     # att, sub_att = data.split('::')
    #     sub_att = data.split(":")[-1].replace(',', '/')
    #     att = sub_att
    #
    #     att = att.strip().split('_')[-1].replace('_', ' ').lower()
    #     sub_att = sub_att.strip().replace('_', ' ').lower().split('(')[0].strip()
    #     att_data[att] = att_data.get(att, [])
    #     att_data[att] += [sub_att]

    # att_data = {k: list(set(v)) for k, v in att_data.items()}
    all_att = {'att': att_data, 'levels': 1}

    json_data = {}
    for k, v in all_att.items():
        if k == 'att':
            json_data['l1_attributes'] = list(v.keys())
        else:
            json_data[k] = v

    json_data['attribute_tree'] = []

    multiprocessing.set_start_method('spawn')

    n_process = 32

    data_slice_list = []
    n_item_per_slice = int(len(att_data) / n_process)
    for i in range(n_process):
        start = i * n_item_per_slice
        end = min(start + n_item_per_slice, len(att_data))
        data_slice_list.append(dic_slice(att_data, start, end))

    process_list = []
    q = multiprocessing.Manager().list()
    for pid in range(n_process):
        slice_datas = data_slice_list[pid]
        print('pid {}'.format(pid))
        process_list.append(
            multiprocessing.Process(target=get_trans, args=(slice_datas, q, json_data['levels'], pid))
        )
    [p.start() for p in process_list]
    [p.join() for p in process_list]

    results = list(q)
    json_data['attribute_tree'] = results
    json.dump(json_data, open(f'../../attributes/{dataset}/{pattern}_.json', 'w'), indent=4)
    json.dump(json_data, open(f'../../attributes/{dataset}/{pattern}.json', 'w'), indent=4, ensure_ascii=False)

