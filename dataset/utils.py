import os
import math
import json
import numpy as np
from collections import defaultdict


def split_uniform(all_data, config):
    dir_path = config['dir_path']
    train_ratio = config['train_ratio']
    client_num = config['client_num']

    np.random.shuffle(all_data)

    size = len(all_data)
    chunk_size = math.ceil(size / client_num)

    for i in range(client_num):
        start = i * chunk_size
        end = min(start + chunk_size, size)
        chunk = all_data[start:end]

        split_idx = int(len(chunk) * train_ratio)
        train_chunk = chunk[:split_idx]
        test_chunk = chunk[split_idx:]

        save_file(train_chunk, f"{dir_path}/train/{i}.jsonl")
        save_file(test_chunk, f"{dir_path}/test/{i}.jsonl")

def split_dir(all_data, config):
    num_clients = config['client_num']
    train_ratio = config['train_ratio']
    dir_path = config['dir_path']
    alpha = config['alpha']
    
    labels = np.array([item["category"] for item in all_data])
    label_set = set(labels)

    min_size = 0
    least_samples = 10
    while min_size < least_samples:
        idx_batch = [[] for _ in range(num_clients)]
        for k in label_set:
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)

            proportions = np.random.dirichlet([alpha] * num_clients)
            split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            for i, s in enumerate(np.split(idx_k, split_points)):
                idx_batch[i].extend(s.tolist())
        min_size = min(len(b) for b in idx_batch)

    os.makedirs(f"{dir_path}/train", exist_ok=True)
    os.makedirs(f"{dir_path}/test", exist_ok=True)

    for j in range(num_clients):
        client_indices = idx_batch[j]
        np.random.shuffle(client_indices)

        split_idx = int(len(client_indices) * train_ratio)
        train_idxs = client_indices[:split_idx]
        test_idxs = client_indices[split_idx:]

        # remove the column "category" before saving
        train_data = [{k: v for k, v in all_data[idx].items() if k != "category"} for idx in train_idxs]
        test_data = [{k: v for k, v in all_data[idx].items() if k != "category"} for idx in test_idxs]
        
        save_file(train_data, f"{dir_path}/train/{j}.jsonl")
        save_file(test_data, f"{dir_path}/test/{j}.jsonl")


def save_file(data, pth):
    import os
    os.makedirs(os.path.dirname(pth), exist_ok=True)
    with open(pth, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
