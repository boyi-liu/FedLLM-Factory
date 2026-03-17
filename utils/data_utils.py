import os

from datasets import load_dataset
from utils.model_utils import load_tokenizer


def get_task_type(model_name):
    if 'robert' in model_name:
        return 'SEQ_CLS'
    if 'bert' in model_name:
        return 'SEQ_CLS'
    if 'qwen3' in model_name:
        return 'CAUSAL_LM'

def load_data(args, idx):
    dataset = args.dataset
    train_dir = os.path.join('dataset', dataset, f'train/{idx}.jsonl')
    test_dir = os.path.join('dataset', dataset, f'test/{idx}.jsonl')

    dataset = load_dataset("json", data_files={'train': train_dir, 'test': test_dir})
    dataset['train'] = dataset['train'].map(get_format_func(args))
    dataset['test'] = dataset['test'].map(get_format_func(args))

    return dataset

def get_format_func(args):
    tokenizer = load_tokenizer(args)
    args.task_type = get_task_type(args.model)

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
            return {
                "input_ids": tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=512
                ).input_ids[0],
                "labels": tokenizer(
                    example["label"],
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=512
                ).input_ids[0]
            }
        return _format_QA