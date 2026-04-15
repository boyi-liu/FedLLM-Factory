import json
import yaml
import os
import numpy as np
from collections import defaultdict
from utils import save_file, split_uniform


if __name__ == "__main__":
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)

    raw_data = []
    for filename in ["train_data_part1.jsonl", "train_data_part2.jsonl", "test_data.jsonl"]:
        with open(f"homebench_raw/{filename}", "r", encoding="utf-8") as f:
            raw_data += [json.loads(line) for line in f]

    all_data = [
        {
            "input_ids": item["input"],
            "label": item["output"],
            "home_id": item["home_id"],
        }
        for item in raw_data
    ]

    mode = config.get("split", "by_home")
    dir_path = config["dir_path"]
    train_ratio = config.get("train_ratio", 0.8)

    os.makedirs(f"{dir_path}/train", exist_ok=True)
    os.makedirs(f"{dir_path}/test", exist_ok=True)

    if mode == "by_home":
        # Each home_id becomes one client (natural non-IID split)
        home_data = defaultdict(list)
        for item in all_data:
            home_data[item["home_id"]].append(item)

        for idx, home_id in enumerate(sorted(home_data.keys())):
            client_data = home_data[home_id]
            np.random.shuffle(client_data)
            split_idx = int(len(client_data) * train_ratio)
            save_file(client_data[:split_idx], f"{dir_path}/train/{idx}.jsonl")
            save_file(client_data[split_idx:], f"{dir_path}/test/{idx}.jsonl")

        print(f"[by_home] Generated {len(home_data)} clients")

    elif mode == "single":
        # Pick the home with the most data, then split uniformly into K clients
        home_data = defaultdict(list)
        for item in all_data:
            home_data[item["home_id"]].append(item)

        top_home = max(home_data, key=lambda h: len(home_data[h]))
        top_data = home_data[top_home]
        print(f"[single] Top home_id: {top_home} ({len(top_data)} samples)")

        split_uniform(top_data, config)
        print(f"[single] Split into {config['client_num']} clients")

    else:
        raise ValueError(f"Unknown mode: {mode!r}. Choose 'by_home' or 'single'.")
