import json
import os

import tqdm
from textblob import TextBlob
from textblob import Word
from textblob.taggers import NLTKTagger
import multiprocessing
import pandas as pd

nltk_tagger = NLTKTagger()


def get_att_categories(pid, path):
    file = path + f'/part-{pid:05d}-5114fd87-297e-42b0-9d11-50f1df323dfa-c000.snappy.parquet'
    if os.path.exists(file.replace('.snappy.parquet', '.json')):
        data = json.load(open(file.replace('.snappy.parquet', '.json'), 'r'))
    else:
        data = list(pd.read_parquet(file)['TEXT'])
        json.dump(data, open(file.replace('.snappy.parquet', '.json'), 'w'), indent=4)
    return_data = {'atts': {}, 'categories': {}}
    for text_str in tqdm.tqdm(data):
        if text_str is None or pd.isna(text_str):
            text_str = ''
        try:
            blob = TextBlob(text_str, pos_tagger=nltk_tagger)
            # print(blob.pos_tags)
            for word, tag in blob.pos_tags:
                if tag == 'JJ':
                    return_data['atts'][word] = return_data['atts'].get(word, 0) + 1
                elif tag == 'NN':
                    return_data['categories'][word] = return_data['categories'].get(word, 0) + 1
                elif tag == 'NNS':
                    word = Word(word).singularize()
                    return_data['categories'][word] = return_data['categories'].get(word, 0) + 1
        except Exception as e:
            print(e)

    json.dump(return_data, open(path + f'/split_{pid}_atts_categories.json', 'w'), indent=4)


def split_json_data(text_list, split_num, path):
    os.makedirs(path, exist_ok=True)
    n_item_per_slice = len(text_list) // split_num
    for i in range(split_num):
        start = i * n_item_per_slice
        end = min(start + n_item_per_slice, len(json_data))
        json.dump(text_list[start: end], open(path+f'/split_{i}.json', 'w'))


def gather_all(path, split_num):
    return_data = {'num_atts': 0, 'num_categories': 0, 'atts': {}, 'categories': {}}
    for i in range(split_num):
        data = json.load(open(path + f'/split_{i}_atts_categories.json', 'r'))
        for k, v in data['atts'].items():
            return_data['atts'][k] = return_data['atts'].get(k, 0) + v
        for k, v in data['categories'].items():
            return_data['categories'][k] = return_data['categories'].get(k, 0) + v

    return_data['atts'] = sorted(return_data['atts'].items(), key=lambda kv: kv[1], reverse=True)
    return_data['categories'] = sorted(return_data['categories'].items(), key=lambda kv: kv[1], reverse=True)
    return_data['num_atts'] = len(return_data['atts'])
    return_data['num_categories'] = len(return_data['categories'])
    return return_data


if __name__ == '__main__':
    # spark = SparkSession.builder.config("spark.driver.memory", "16G").master("local[8]").appName(
    #     'spark-stats').getOrCreate()
    # df = spark.read.parquet("laion2B")
    #
    # df.filter((df.width >= 1024) & (df.height >= 1024))
    # df = df.orderBy(rand())  # this line is important to have a shuffled dataset
    #
    # df.repartition(128).write("laion2B_big")

    multiprocessing.set_start_method('spawn')
    n_process = 127

    process_list = []
    for pid in range(n_process):
        print('pid {}'.format(pid))
        process_list.append(
            multiprocessing.Process(target=get_att_categories, args=(pid, '/data1/kyanchen/prompt/data/laion2b'))
        )
    [p.start() for p in process_list]
    [p.join() for p in process_list]

    return_data = gather_all(path='/data1/kyanchen/prompt/data/laion2b', split_num=n_process)
    return_atts = {'num_atts': return_data['num_atts'], 'atts': return_data['atts']}
    return_objects = {'num_categories': return_data['num_categories'], 'categories': return_data['categories']}
    json.dump(return_atts, open(f'extracted_atts.json', 'w'), indent=4)
    json.dump(return_objects, open(f'extracted_categories.json', 'w'), indent=4)
