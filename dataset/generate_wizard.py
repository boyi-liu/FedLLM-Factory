import json
import yaml
import os
import urllib.request

from dataset.rag.precess_rag import process_rag
from ft.utils import split_dataset



if __name__ == "__main__":
    # === Load configuration file ===
    with open("config.yaml", "r") as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)

    # === Define dataset path ===
    dataset_dir = "wizard"
    dataset_path = os.path.join(dataset_dir, "WizardLM_evol_instruct_V2_143k.json")

    # === Load dataset ===
    print("Loading dataset...")
    data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        try:
            # Try to parse as JSON array
            data = json.loads(content)
        except json.JSONDecodeError:
            # If that fails, try JSONL format
            data = [json.loads(line) for line in content.splitlines() if line.strip()]

    print(f"Loaded {len(data)} samples from Wizard dataset.")

    # === Split into train/test sets ===
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
