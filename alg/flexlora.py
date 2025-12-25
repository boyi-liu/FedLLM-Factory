import torch

from alg.ftbase import FTBaseClient, FTBaseServer
from utils.time_utils import time_record


def add_args(parser):
    parser.add_argument('--s', type=float, default=2, help="Scaling factor")
    return parser.parse_args()

class Client(FTBaseClient):
    @time_record
    def run(self, model):
        super().run(model)

class Server(FTBaseServer):
    def __init__(self, args, clients):
        super().__init__(args, clients)
        self.r = args.lora_rank
        self.s = args.s

    def run(self):
        self.sample()
        self.local_run()
        self.aggregate()

    def aggregate(self):
        data_sum = sum([len(client.dataset['train']) for client in self.sampled_clients])

        from collections import defaultdict
        aggregated = defaultdict(lambda: 0)

        lora_keys = [k for k in self.global_lora.keys() if 'lora_A' in k]

        for a_key in lora_keys:
            b_key = a_key.replace('lora_A', 'lora_B')
            base_key = a_key.replace('.lora_A.default.weight', '.base_layer.weight')
            a_list = []
            b_list = []

            for client in self.sampled_clients:
                p_k = len(client.dataset['train']) / data_sum
                a_list.append(p_k * client.lora[a_key])
                b_list.append(client.lora[b_key])

            stacked_A = torch.cat(a_list, dim=0)
            stacked_B = torch.cat(b_list, dim=1)
            delta_W = torch.matmul(stacked_B, stacked_A) * self.s

            U, Sigma, V = torch.linalg.svd(delta_W.float(), full_matrices=False)
            Sigma_truncated = Sigma[:self.r]
            sqrt_Sigma = torch.diag(torch.sqrt(Sigma_truncated))
            # print('U.shape', U.shape)
            # print('Sigma.shape', Sigma.shape)
            # print('V.shape', V.shape)

            aggregated[b_key] = torch.matmul(U[:, :self.r], sqrt_Sigma) / self.s
            aggregated[a_key] = V[:self.r, :]

        self.global_lora = aggregated
        self.model.load_state_dict(self.global_lora, strict=False)
        print("Aggregated model updated.")