import json
import yaml
import os
import numpy as np
from collections import defaultdict
from utils import save_file, split_uniform


if __name__ == "__main__":
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)

    with open("sensorqa_raw/overall_sensorqa_dataset_train.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    with open("sensorqa_raw/overall_sensorqa_dataset_val.json", "r", encoding="utf-8") as f:
        raw_data += json.load(f)

    all_data = [
        {
            "input_ids": item["question"],
            "label": item["answer"],
            "user": item["user"],
        }
        for item in raw_data
    ]

    mode = config.get("split", "by_user")
    
    dir_path = config["dir_path"]
    train_ratio = config.get("train_ratio", 0.8)

    os.makedirs(f"{dir_path}/train", exist_ok=True)
    os.makedirs(f"{dir_path}/test", exist_ok=True)

    if mode == "by_user":
        # Each user becomes one client (natural non-IID split)
        user_data = defaultdict(list)
        for item in all_data:
            user_data[item["user"]].append(item)

        for idx, user in enumerate(sorted(user_data.keys())):
            client_data = user_data[user]
            np.random.shuffle(client_data)
            split_idx = int(len(client_data) * train_ratio)
            save_file(client_data[:split_idx], f"{dir_path}/train/{idx}.jsonl")
            save_file(client_data[split_idx:], f"{dir_path}/test/{idx}.jsonl")

        print(f"[by_user] Generated {len(user_data)} clients")

    elif mode == "single":
        # Pick the user with the most data, then split uniformly into K clients
        user_data = defaultdict(list)
        for item in all_data:
            user_data[item["user"]].append(item)

        top_user = max(user_data, key=lambda u: len(user_data[u]))
        top_data = user_data[top_user]
        print(f"[single] Top user: {top_user} ({len(top_data)} samples)")

        split_uniform(top_data, config)
        print(f"[single] Split into {config['client_num']} clients")

    else:
        raise ValueError(f"Unknown mode: {mode!r}. Choose 'by_user' or 'single'.")
