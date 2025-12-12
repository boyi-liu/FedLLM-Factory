import numpy as np
import torch
import cvxpy as cp
import copy

from alg.ftbase import FTBaseClient, FTBaseServer
from utils.time_utils import time_record

def add_args(parser):
    parser.add_argument('--alpha', type=float, default=1.5, help="Alpha in weight optimization")
    parser.add_argument('--lam', type=float, default=0.01, help="Lambda in local training")
    return parser.parse_args()


class Client(FTBaseClient):
    @time_record
    def run(self, model):
        super().run(model)


class Server(FTBaseServer):
    def __init__(self, args, clients):
        super().__init__(args, clients)
        client_num = len(clients)
        self.sims = np.zeros([client_num, client_num])
        self.graph_w = np.zeros([client_num, client_num])
        self.client_loras = []  # 保存每个 client 自适应聚合后的 LoRA
        self.alpha = args.alpha
        

    def run(self):
        self.sample()
        self.local_run()
        self.update_sims()
        self.update_w()
        self.aggregate()

    def p_model(self, c_id):
        c_lora = self.client_loras[c_id]
        c_model = copy.deepcopy(self.model)
        c_model.load_state_dict(c_lora, strict=False)
        return c_model

    def local_run(self):
        for client in self.sampled_clients:
            client.run(self.p_model(client.id))
        self.wall_clock_time += max([c.training_time for c in self.sampled_clients])

    def update_sims(self):
        def state_dict_to_tensor(state_dict):
            return torch.cat([v.flatten() for v in state_dict.values()])

        for c_i in self.sampled_clients:
            for c_j in self.clients:
                idx_i = c_i.id
                idx_j = c_j.id

                lora_i = state_dict_to_tensor(c_i.lora)
                lora_j = state_dict_to_tensor(c_j.lora)
                
                if lora_i.numel() == 0 or lora_j.numel() == 0:
                    continue
                    
                self.sims[idx_i, idx_j] = self.sims[idx_j, idx_i] = torch.nn.functional.cosine_similarity(
                    lora_i.unsqueeze(0),
                    lora_j.unsqueeze(0),
                    dim=1).item()

    # https://github.com/MediaBrain-SJTU/pFedGraph/blob/main/pfedgraph_cosine/utils.py
    def update_w(self):
        total_samples = sum(len(client.dataset_train) for client in self.clients)
        w_all = [len(c.dataset_train) / total_samples for c in self.clients]

        for idx, c in enumerate(self.sampled_clients):
            sims = self.sims[c.id]
            n = len(self.clients)
            p = np.array(w_all)
            P = np.identity(n)
            P = cp.atoms.affine.wraps.psd_wrap(P)
            G = - np.identity(n)
            h = np.zeros(n)
            A = np.ones((1, n))
            b = np.ones(1)
            d = sims
            q = - d * self.alpha - 2 * p
            x = cp.Variable(n)
            prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + q.T @ x),
                              [G @ x <= h,
                               A @ x == b])
            prob.solve()
            self.graph_w[idx] = torch.Tensor(x.value)

    def aggregate(self):
        lora_keys = None
        for client in self.clients:
            if hasattr(client, "lora") and client.lora:
                lora_keys = list(client.lora.keys())
                break

        aggregated_list = []
        for c in self.clients:
            weights = self.graph_w[c.id]  # shape: [num_clients]
            agg_lora = {}
            for k in lora_keys:
                acc = None
                for j, other in enumerate(self.clients):
                    w = torch.tensor(weights[j])
                    term = w * other.lora[k]
                    acc = term if acc is None else acc + term
                agg_lora[k] = acc
            aggregated_list.append(agg_lora)

        self.client_loras = aggregated_list

    def test_all(self):
        all_metrics = []
        for client in self.clients:
            print(f"Testing on client {client.id} ...")
            metrics = client.local_test(self.p_model(client.id))
            all_metrics.append(metrics)

        avg_loss = sum(m["eval_loss"] for m in all_metrics) / len(all_metrics)
        avg_perplexity = sum(m["perplexity"] for m in all_metrics) / len(all_metrics)
        return {'loss': avg_loss, 'perplexity': avg_perplexity}