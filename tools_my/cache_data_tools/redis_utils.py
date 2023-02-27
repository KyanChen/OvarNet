
"""
为训练中从Redis存取数据，初始化Redis，清空Redis等提供接口支持
"""
from ast import arg
from multiprocessing import pool

import mmcv
import numpy as np
import redis
try:
    import ujson as json
except:
    import json
import tqdm
import os
import sys
import random
from functools import wraps
import pandas as pd

def singleton(cls):
    _instance = {}
    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]
    return inner

def reconnect_decorator(func):
    def wrapper(self,*args,**kwargs):
        try:
            res = func(self,*args,**kwargs)
        except Exception as e:
            print(func,e,'reconnect redis')
            self.init_redis()
            res = func(self,*args,**kwargs)
        return res
    return wrapper


class RedisHelper(object):

    def __init__(self):
        self.redis = None
        self._pool = None
       
    def init_redis(self):
        """初始化redis链接，用于lazy init
        """
        self.redis = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True) 
    
    @reconnect_decorator
    def get_item_key(self, dataset_name_wo_suffix, note_id):
        """获取一个item的唯一id
        """
        return f'{dataset_name_wo_suffix}_{note_id}'

    @reconnect_decorator
    def clear_all(self):
        """清除所有redis key
        """
        print('clear {} keys ...'.format(len(self.redis.keys())))
        [self.redis.delete(x) for x in tqdm.tqdm(self.redis.keys())]
        return

    def save_all_data(self, pkl_file):
        all_data = {}
        for x in tqdm.tqdm(self.redis.keys()):
            all_data[x] = self.get_values(x)
        mmcv.dump(all_data, pkl_file)

    @reconnect_decorator
    def get_values(self, key):
        """根据图像id获取图像信息
        """ 
        if not self.redis:
            self.init_redis()
        
        img_info = json.loads(self.redis.get(key))
        return img_info

    @reconnect_decorator
    def set_values(self, key, value):
        """根据图像id获取图像信息
        """
        if not self.redis:
            self.init_redis()
        self.redis.set(key, json.dumps(value))
    
    def init_dataset_memory_with_CSVs(self, csv_infos_list):
        # import pdb
        # pdb.set_trace()
        for csv_infos in csv_infos_list:
            assert set(csv_infos.keys()) >= set(['csv_file', 'text_file', 'keys_map'])
            self.init_dataset_memory_with_CSV(**csv_infos)

    def init_dataset_memory_with_CSV(
        self, 
        csv_file, 
        text_file,
        csv_header="infer",
        keys_map={
                    'note_id': 'discovery_id',
                    'label': 'review_mark',
                    'title': 'title', 
                    'content': 'content',
                    'urls': 'file_url_list'
                },
        collect_keys=['note_id', 'urls', 'label', 'title', 'content', 'text_embedding'],
        replace_kv=True,
        *args, **kwargs
        ):
        """ 将dataset 写入redis 
        """
        if not self.redis:
            self.init_redis()
        print('start load dataset {} to redis'.format(csv_file))

        #如果已经导入过，那么返回
        dataset_name_wo_suffix = os.path.basename(csv_file).split('.')[0]
        if self.redis.get(dataset_name_wo_suffix):
            print('dataset {} has aleady loaded in redis'.format(csv_file))
            # return 
        
        data = pd.read_csv(csv_file, header=csv_header)

        if os.path.splitext(text_file)[1] == '.txt':
            texts = np.loadtxt(text_file, dtype=np.float32, ndmin=2, delimiter=',')
            # texts = open(text_file, 'r').readlines()
            # texts = [np.array(x.strip().split(','), dtype=np.float32) for x in texts]
        elif os.path.splitext(text_file)[1] == '.npy':
            texts = np.load(text_file)
        print('total embedding: ', texts.shape)
        assert len(texts) == len(data)

        for idx, rows_data in tqdm.tqdm(data.iterrows()):
            note_id = rows_data[keys_map['note_id']]
            # if self.redis.get(note_id):
            #     continue

            urls = rows_data[keys_map['urls']]
            if pd.isnull(urls):
                urls = ''
            urls = urls.replace('[', '').replace(']', '').replace('\"', '').split(',')
            # urls = ['http://xhsci-qn.xiaohongshu.com/'+x for x in urls if len(x) > 5]
            if len(urls) == 0:
                urls = ['']
            
            title = rows_data[keys_map['title']]
            if pd.isnull(title):
                title = ''

            content = rows_data[keys_map['content']]
            if pd.isnull(content):
                content = ''
            
            # if isinstance(rows_data[keys_map['label']], float):
            #     print
            #     label = repr(rows_data[keys_map['label']])
            # else:
            try:
                label = rows_data[keys_map['label']].replace('\"', '').strip()
            except:
                print(label)
                label = '不推荐'
            # label = '不推荐'

            txt_embed = texts[idx].tolist()
            if len(title+content) == 0:
                txt_embed = [0] * len(txt_embed)
            
           
            write_key = f'{note_id}'
            write_tmp_value = {
                'note_id': note_id,
                'urls': urls,
                'title': title,
                'content': content,
                'label': label,
                'text_embedding': txt_embed
            }

            write_value = {k: v for k, v in write_tmp_value.items() if k in collect_keys}
            
            if not self.redis.get(write_key) or replace_kv:
                self.redis.set(write_key, json.dumps(write_value))

        print('dataset: {}, data count: {}'.format(dataset_name_wo_suffix, len(data)))
        #存储数据集基本信息
        self.redis.set(dataset_name_wo_suffix, 1)

        return len(data)

    def init_dataset_memory_with_json(
            self,
            json_file,  # file or json_data
            json_data=None,
            prefix='cky_',
            *args, **kwargs
    ):
        """ 将dataset 写入redis
        """
        if not self.redis:
            self.init_redis()
        dataset_name_wo_suffix = os.path.basename(json_file).split('.')[0]
        print('start load dataset {} to redis'.format(json_file))
        if json_data is not None:
            data = json_data
        else:
            data = json.load(open(json_file, 'r'))
        for img_id, v in tqdm.tqdm(data.items()):
            v['img_id'] = img_id
            img_id = prefix + str(img_id)
            self.redis.set(img_id, json.dumps(v))

        print('dataset: {}, data count: {}'.format(json_file, len(data)))
        # 存储数据集基本信息
        self.redis.set(dataset_name_wo_suffix, 1)
        return len(data)


if __name__ == '__main__':
    redis_helper = RedisHelper()
    # redis_helper.init_dataset_memory_with_CSVs()
    # redis_helper.init_dataset_memory_with_json()
    redis_helper.save_all_data('train2017_proposals_predatts.pkl')
    
        
