import json
import yaml

from utils import split_dataset

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)

    with open("gsm8k/train.jsonl", "r", encoding="utf-8") as f:
        train_data = [json.loads(line) for line in f]
    with open("gsm8k/test.jsonl", "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f]

    split_dataset(config, {'train': train_data, 'test': test_data})
