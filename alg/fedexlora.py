import torch

from alg.ftbase import FTBaseClient, FTBaseServer
from utils.time_utils import time_record

class Client(FTBaseClient):
    @time_record
    def run(self, model):
        super().run(model)

class Server(FTBaseServer):
    def run(self):
        self.sample()
        self.local_run()
        self.aggregate()

    def aggregate(self):
        data_total = sum([len(client.dataset['train']) for client in self.sampled_clients])

        lora_keys = [k for k in self.global_lora.keys() if 'lora_A' in k]

        from collections import defaultdict

        for a_key in lora_keys:
            b_key = a_key.replace('lora_A', 'lora_B')
            base_key = a_key.replace('.lora_A.default.weight', '.base_layer.weight')
            a_list = []
            b_list = []
            p_list = []

            for client in self.sampled_clients:
                p_k = len(client.dataset['train']) / data_total
                a_list.append(client.lora[a_key])
                b_list.append(client.lora[b_key])
                p_list.append(p_k)
            p_tensor = torch.tensor(p_list).to(a_list[-1].device).view(-1, 1, 1)

            A = torch.stack(a_list)
            B = torch.stack(b_list)

            term1 = torch.sum(p_tensor * (B @ A), dim=0)
            term2 = torch.mean(B, dim=0) @ torch.mean(A, dim=0)
            res = term1 - term2
            self.model.state_dict()[base_key].data.add_(res)

            self.global_lora[a_key] = torch.mean(A, dim=0)
            self.global_lora[b_key] = torch.mean(B, dim=0)

        self.model.load_state_dict(self.global_lora, strict=False)
        print("Aggregated model updated.")