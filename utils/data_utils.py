import os

from datasets import load_dataset


def load_data(args, idx):
    dataset = args.dataset
    train_dir = os.path.join('./dataset', dataset, f'train/{idx}.json')
    test_dir = os.path.join('./dataset', dataset, f'test/{idx}.json')

    dataset = load_dataset("json", data_files={'train': train_dir, 'test': test_dir})
    dataset['train'] = dataset['train'].map(format_QA)
    dataset['test'] = dataset['test'].map(format_QA)

    return dataset

def format_QA(self, example):
    prompt = f"Instruct: {example['question']}\nAnswer:"
    return {
        "input_ids": self.tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=512).input_ids[0],
        "labels": self.tokenizer(example["answer"], return_tensors="pt", truncation=True, padding="max_length", max_length=512).input_ids[0]
    }