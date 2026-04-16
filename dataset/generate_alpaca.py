import json
import yaml
from utils import split_dir


def categorize(instruction: str) -> str:
    inst = instruction.lower()
    if any(inst.startswith(w) for w in ("write", "create", "generate", "compose", "draft")):
        return "creative_writing"
    if any(inst.startswith(w) for w in ("translate", "convert", "transform", "rewrite")):
        return "transformation"
    if any(inst.startswith(w) for w in ("list", "give", "provide", "name", "identify")):
        return "brainstorming"
    if any(inst.startswith(w) for w in ("explain", "describe", "what is", "what are", "define")):
        return "information"
    if any(inst.startswith(w) for w in ("how", "why", "when", "where", "who")):
        return "open_qa"
    if any(inst.startswith(w) for w in ("classify", "categorize", "determine", "evaluate", "judge")):
        return "classification"
    if any(inst.startswith(w) for w in ("summarize", "summarise", "condense", "shorten")):
        return "summarization"
    return "general_qa"


if __name__ == "__main__":
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)

    with open("alpaca_raw/alpaca_data_cleaned.json", "r", encoding="utf-8") as f:
        all_data = json.load(f)

    merged_all_data = []
    for data in all_data:
        merged_all_data.append(
            {
                "input_ids": f'Instruction: {data["instruction"]}\nInput: {data["input"]}',
                "label": data["output"],
                "category": categorize(data["instruction"])
            }
        )

    split_dir(merged_all_data, config)
