import torch

from alg.ftbase import FTBaseClient, FTBaseServer
from utils.time_utils import time_record

class Client(FTBaseClient):
    @time_record
    def run(self, model):
        client_model = model

        # freeze all parameters first
        for param in client_model.parameters():
            param.requires_grad = False

        # then unfreeze lora_B
        for name, param in client_model.named_parameters():
            if "lora_B" in name:
                param.requires_grad = True

        self.trainer.train(client_model)
        self.lora = {k: v.clone() for k, v in client_model.state_dict().items()}

class Server(FTBaseServer):
    def __init__(self, args, clients):
        super().__init__(args, clients)

    def run(self):
        self.sample()
        self.local_run()
        self.aggregate()

    def aggregate(self):
        r = self.args.lora_rank
        
        # 1. Aggregate LoRA_B First
        data_sum = sum([len(client.dataset['train']) for client in self.sampled_clients])
        from collections import defaultdict
        aggregated = defaultdict(lambda: 0)

        for client in self.sampled_clients:
            model = client.lora
            for k, v in model.items():
                if 'lora_B' in k: aggregated[k] = aggregated[k] + v * len(client.dataset['train']) / data_sum

        # 2. Generate SVD Decomposition for LoRA_A
        lora_keys = [k for k in self.global_lora.keys() if 'lora_A' in k]
        for a_key in lora_keys:
            b_key = a_key.replace('lora_A', 'lora_B')
            
            A = self.global_lora[a_key]
            B = aggregated[b_key]
            U, S, V = torch.svd(B @ A)
            self.global_lora[a_key] = V.t()[:r, :]
            self.global_lora[b_key] = U[:, :r] @ torch.diag(S[:r])
            
        self.model.load_state_dict(self.global_lora, strict=False)
        print("Aggregated model updated.")
