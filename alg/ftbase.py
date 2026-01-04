import os
import random

from peft import get_peft_model

from utils.sys_utils import device_config
from utils.train_utils import Trainer
from alg.base import BaseClient, BaseServer
from datasets import load_dataset
from utils.model_utils import load_model, load_tokenizer, load_lora_config
from utils.time_utils import time_record


class FTBaseClient(BaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.tokenizer = load_tokenizer(args)
        self.load_data()
        self.lora = {}
        self.trainer = Trainer(args=args, dataset=self.dataset, client=self)

    def load_data(self):
        train_dir = os.path.join('./dataset', self.args.dataset, f'train/{self.id}.json')
        test_dir = os.path.join('./dataset', self.args.dataset, f'test/{self.id}.json')

        self.dataset = load_dataset("json", data_files={'train': train_dir, 'test': test_dir})
        self.dataset['train'] = self.dataset['train'].map(self.format_example)
        self.dataset['test'] = self.dataset['test'].map(self.format_example)

    def format_example(self, example):
        prompt = f"Instruct: {example['question']}\nAnswer:"
        return {
            "input_ids": self.tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=512).input_ids[0],
            "labels": self.tokenizer(example["answer"], return_tensors="pt", truncation=True, padding="max_length", max_length=512).input_ids[0]
        }

    @time_record
    def run(self, model):
        print(f'\nClient {self.id} starting...')
        self.trainer.train(model)
        self.lora = {k: v.clone() for k, v in model.state_dict().items() if "lora_" in k}

    def local_test(self, model):
        return self.trainer.eval(model)

class FTBaseServer(BaseServer):
    def __init__(self, args, clients):
        super().__init__(args, clients)
        self.model = get_peft_model(load_model(args), load_lora_config(args))

        self.global_lora = {k: v.clone() for k, v in self.model.state_dict().items() if "lora_" in k}
        self.sample_rate = args.sr
        self.wall_clock_time = 0
        self.round = 0

        for client, delay in zip(clients, device_config(args)): client.delay = delay

    def run(self):
        self.sample()
        self.local_run()
        self.aggregate()

    def sample(self):
        sample_num = int(self.sample_rate * len(self.clients))
        self.sampled_clients = sorted(random.sample(self.clients, sample_num), key=lambda x: x.id)

    def local_run(self):
        for client in self.sampled_clients:
            client.run(self.model)
            self.model.load_state_dict(self.global_lora, strict=False)
        self.wall_clock_time += max([c.training_time for c in self.sampled_clients])

    def aggregate(self):
        data_sum = sum([len(client.dataset['train']) for client in self.sampled_clients])
        from collections import defaultdict
        aggregated = defaultdict(lambda: 0)

        for client in self.sampled_clients:
            model = client.lora
            for k, v in model.items():
                aggregated[k] = aggregated[k] + v * len(client.dataset['train']) / data_sum

        self.global_lora = aggregated
        self.model.load_state_dict(self.global_lora, strict=False)
        print("Aggregated model updated.")

    def test_all(self):
        all_metrics = []
        for client in self.clients:
            print(f"Testing on client {client.id} ...")
            metrics = client.local_test(self.model)
            all_metrics.append(metrics)

        avg_loss = sum(m["eval_loss"] for m in all_metrics) / len(all_metrics)
        avg_perplexity = sum(m["perplexity"] for m in all_metrics) / len(all_metrics)
        return {'loss': avg_loss, 'perplexity': avg_perplexity}