import json
import yaml
from utils import split_dir

if __name__ == "__main__":
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)

    with open("dolly_raw/databricks-dolly-15k.jsonl", "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f]
        
    merged_all_data = []
    for data in all_data:
        merged_all_data.append(
            {
                "input_ids": f'Instruction: {data["instruction"]}\nContext: {data["context"]}',
                "label": data["response"],
                "category": data["category"]
            }
        )
        
    split_dir(merged_all_data, config)