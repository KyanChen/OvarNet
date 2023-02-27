import json

parent_dir = '../../attributes/LSA'
datasets = ['test.json']
infos = {}
for dataset in datasets:
    data = json.load(open(parent_dir+'/'+dataset, 'r'))
    for item in data:
        img_id = item['image_id']
        objects = item['objects']

        dataset_key = img_id.split('_')[0]
        infos[dataset_key] = infos.get(dataset_key, {})
        infos[dataset_key]['num_img'] = infos[dataset_key].get('num_img', 0) + 1
        for one_object in objects:
            infos[dataset_key]['object'] = infos[dataset_key].get('object', set())
            infos[dataset_key]['object'].add(one_object['object'])
            for att in one_object['attributes']:
                infos[dataset_key]['attributes'] = infos[dataset_key].get('attributes', set())
                infos[dataset_key]['attributes'].add(att)
            infos[dataset_key]['num_instance'] = infos[dataset_key].get('num_instance', 0) + 1
            infos[dataset_key]['ground'] = infos[dataset_key].get('ground', set())
            infos[dataset_key]['ground'].add(one_object['ground'])
    extra_info = {}
    for k, v in infos.items():
        extra_info[k + '_num_object'] = len(v['object'])
        extra_info[k + '_num_attributes'] = len(v['attributes'])
    for k, v in infos.items():
        extra_info['all_objects'] = extra_info.get('all_objects', set())
        extra_info['all_objects'] = extra_info['all_objects'] | v['object']
        extra_info['all_attributes'] = extra_info.get('all_attributes', set())
        extra_info['all_attributes'] = extra_info['all_attributes'] | v['attributes']
    extra_info['all_num_object'] = len(extra_info['all_objects'])
    extra_info['all_num_attributes'] = len(extra_info['all_attributes'])
    infos.update(extra_info)
    infos_write = {}
    for k, v in infos.items():
        if isinstance(v, dict):
            for k1, v1 in v.items():
                infos_write[k] = infos_write.get(k, {})
                infos_write[k][k1] = list(v1) if isinstance(v1, set) else v1
        else:
            infos_write[k] = list(v) if isinstance(v, set) else v

    for k, v in infos_write.items():
        if isinstance(v, dict):
            for k1, v1 in v.items():
                if k1 in ['object', 'attributes']:
                    infos_write[k][k1] = []
        if k in ['all_objects', 'all_attributes']:
            infos_write[k] = []

    json.dump(infos_write, open(parent_dir+'/'+dataset.replace('.json', '_simple_infos.json'), 'w'), indent=4)


