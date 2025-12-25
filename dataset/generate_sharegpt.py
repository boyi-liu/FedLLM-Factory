import os
import json
import yaml
from datasets import load_dataset

from dataset.rag.precess_rag import process_rag
from ft.utils import split_dataset


def load_sharegpt_dataset(local_path="sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json"):
    """
    Try to load ShareGPT dataset.
    If local file exists, read it.
    Otherwise, attempt to load it from Hugging Face.
    """
    os.makedirs("sharegpt", exist_ok=True)

    if os.path.exists(local_path):
        print(f"Found local dataset: {local_path}")
        with open(local_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        print("Loading ShareGPT dataset from Hugging Face ...")
        try:
            dataset = load_dataset("ShareGPT/sharegpt_vicuna", split="train")
            data = dataset.to_list()
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Downloaded and saved dataset to {local_path}")
        except Exception as e:
            print("Failed to load dataset automatically.")
            print("Please download it manually from:")
            print("https://huggingface.co/datasets/ShareGPT/sharegpt_vicuna")
            raise e

    return data


if __name__ == "__main__":
    # === Load configuration ===
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)

    # === Load ShareGPT dataset ===
    data = load_sharegpt_dataset()

    # === Split data into train/test ===
    split_ratio = 0.9
    split_index = int(len(data) * split_ratio)
    train_data = data[:split_index]
    test_data = data[split_index:]

    print(f"Total samples: {len(data)} | Train: {len(train_data)} | Test: {len(test_data)}")

    # === Choose processing type ===
    if config["type"] == "ft":
        split_dataset(config, {"train": train_data, "test": test_data})
    else:
        process_rag(config)
