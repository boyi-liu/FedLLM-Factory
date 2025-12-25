import json
import yaml
import os
import urllib.request

from dataset.rag.precess_rag import process_rag
from ft.utils import split_dataset


def download_dataset(url, save_path):
    """Download the dataset from a URL if it does not exist locally."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(save_path):
        print(f"Downloading CodeAlpaca dataset from {url} ...")
        urllib.request.urlretrieve(url, save_path)
        print("Download completed.")
    else:
        print("Dataset already exists, skipping download.")


if __name__ == "__main__":
    # === Load configuration ===
    with open("config.yaml", "r") as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)

    # === Dataset path ===
    dataset_dir = "codealpaca"
    dataset_path = os.path.join(dataset_dir, "code_alpaca_20k.json")

    # === Download dataset if not found ===
    dataset_url = "https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k/resolve/main/code_alpaca_20k.json"
    download_dataset(dataset_url, dataset_path)

    # === Load dataset ===
    print("Loading dataset...")
    data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    # Handle the case where the file is a pure JSON array
                    f.seek(0)
                    data = json.load(f)
                    break

    print(f"Loaded {len(data)} samples from CodeAlpaca dataset.")

    # === Split into train/test ===
    split_ratio = 0.9
    split_index = int(len(data) * split_ratio)
    train_data = data[:split_index]
    test_data = data[split_index:]

    # === Process according to config ===
    if config["type"] == "ft":
        print("Running fine-tuning mode...")
        split_dataset(config, {"train": train_data, "test": test_data})
    else:
        print("Running retrieval-augmented generation (RAG) mode...")
        process_rag(config)

    print("Processing finished successfully.")
