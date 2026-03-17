import os
import math
import json


def split_uniform(data, dir_path, client_num):
    os.makedirs(dir_path, exist_ok=True)
    size = len(data)
    chunk_size = math.ceil(size / client_num)

    for i in range(client_num):
        start = i * chunk_size
        end = min(start + chunk_size, size)
        chunk = data[start:end]
        if not chunk: break

        out_file = f"{dir_path}/{i}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            for item in chunk:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


def split_dataset(cfg, dataset):
    client_num = cfg['client_num']
    dir_path = cfg['dir_path']
    os.makedirs(dir_path, exist_ok=True)

    if cfg['split'] == 'uniform':
        split_uniform(dataset['train'], f'{dir_path}/train', client_num)
        split_uniform(dataset['test'], f'{dir_path}/test', client_num)
