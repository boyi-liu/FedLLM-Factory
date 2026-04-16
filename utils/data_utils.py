import os

from datasets import load_dataset
from utils.model_utils import load_tokenizer

def load_data(args, idx):
    dataset = args.dataset
    train_dir = os.path.join('dataset', dataset, f'train/{idx}.jsonl')
    test_dir = os.path.join('dataset', dataset, f'test/{idx}.jsonl')

    dataset = load_dataset("json", data_files={'train': train_dir, 'test': test_dir})
    tokenizer = load_tokenizer(args)
    format_func = get_format_func(args, tokenizer)
    dataset['train'] = dataset['train'].map(format_func)
    dataset['test'] = dataset['test'].map(format_func)

    return dataset

def get_format_func(args, tokenizer):
    if args.task_type == 'SEQ_CLS':
        def _format_classification(example):
            tokenized = tokenizer(
                    example["input_ids"],
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=512
                )
            return {
                "input_ids": tokenized.input_ids[0],
                "attention_mask": tokenized.attention_mask[0],
                "labels": example["label"]
            }
        return _format_classification
    elif args.task_type == 'CAUSAL_LM':
        def _format_QA(example):
            prompt = f"Instruct: {example['input_ids']}\nAnswer:"
            full_text = prompt + example["label"]
            input_ids = tokenizer(
                full_text,
                truncation=True,
                padding="max_length",
                max_length=512
            )["input_ids"]
            prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            pad_id = tokenizer.pad_token_id
            labels = [-100] * prompt_len + [
                t if t != pad_id else -100 for t in input_ids[prompt_len:]
            ]
            return {
                "input_ids": input_ids,
                "labels": labels
            }
        return _format_QA