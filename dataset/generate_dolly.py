import json
import yaml
from dataset.rag.precess_rag import process_rag
from ft.utils import split_dataset
import random

if __name__ == "__main__":
    # === 1. Load configuration ===
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)

    # === 2. Load Dolly dataset ===
    dataset_path = "dolly15k/databricks-dolly-15k.json"

    data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # skip empty lines
                data.append(json.loads(line))

    # === 3. Shuffle and split into train/test sets ===
    random.shuffle(data)
    split_index = int(0.9 * len(data))
    train_data = data[:split_index]
    test_data = data[split_index:]

    # === 4. Normalize sample format ===
    def normalize_dolly(example):
        return {
            "instruction": example.get("instruction", ""),
            "input": example.get("context", ""),
            "output": example.get("response", "")
        }

    train_data = [normalize_dolly(d) for d in train_data]
    test_data = [normalize_dolly(d) for d in test_data]

    # === 5. Run data processing ===
    if config["type"] == "ft":
        # Fine-tuning mode
        split_dataset(config, {'train': train_data, 'test': test_data})
    else:
        # RAG mode
        process_rag(config)

    print(f"Dolly dataset processed successfully.")
    print(f"Total samples: {len(data)} | Train: {len(train_data)} | Test: {len(test_data)}")
