import yaml

from datasets import load_dataset
from utils import split_dir

dataset = load_dataset("glue", "sst2", cache_dir='./sst2_raw')

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)

    all_data = []
    for split in ['train', 'validation']:
        for item in dataset[split]:
            all_data.append({"input_ids": item["sentence"], 
                             "label": item["label"],
                             "category": item["label"]
                             }) 

    split_dir(all_data, config)