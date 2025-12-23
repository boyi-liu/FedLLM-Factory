import torch

from alg.ftbase import FTBaseClient, FTBaseServer
from utils.time_utils import time_record

class Client(FTBaseClient):
    @time_record
    def run(self, model):
        return super().run(model)

class Server(FTBaseServer):
    def __init__(self, args, clients):
        super().__init__(args, clients)

    def run(self):
        self.sample()
        self.local_run()
        self.aggregate()

    def aggregate(self):
        data_total = sum([len(client.dataset['train']) for client in self.sampled_clients])

        lora_keys = [k for k in self.global_lora.keys() if 'lora_A' in k]

        for a_key in lora_keys:
            b_key = a_key.replace('lora_A', 'lora_B')
            base_key = a_key.replace('.lora_A.default.weight', '.base_layer.weight')
            a_list = []
            b_list = []

            for client in self.sampled_clients:
                p_k = len(client.dataset['train']) / data_total
                a_list.append(p_k * client.lora[a_key])
                b_list.append(client.lora[b_key])

            stacked_A = torch.cat(a_list, dim=0)
            stacked_B = torch.cat(b_list, dim=1)
            delta_W = torch.matmul(stacked_B, stacked_A)
            self.model.state_dict()[base_key].data.add_(delta_W)