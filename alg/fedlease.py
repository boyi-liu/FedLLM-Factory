import torch
import numpy as np
import torch.nn as nn

from peft.tuners.lora import LoraLayer
from alg.ftbase import FTBaseClient, FTBaseServer
from utils.time_utils import time_record


class FedLEASEMoELayer(nn.Module):
    def __init__(self, peft_layer: LoraLayer,
                 num_experts: int):
        super().__init__()

        # 1. Steal the Base Layer
        # peft_layer.base_layer is the original layer of the model (frozen)
        self.base_layer = peft_layer.base_layer
        self.peft_layer = peft_layer
        self.in_features = self.base_layer.in_features
        self.out_features = self.base_layer.out_features

        # 2. Get the configuration (reuse the peft configuration)
        adapter_name = peft_layer.active_adapter[0]  # Usually 'default'
        r = peft_layer.r[adapter_name] 
        alpha = peft_layer.lora_alpha[adapter_name]
        dropout = peft_layer.lora_dropout[adapter_name].p
        scaling = alpha / r

        # 3. Initialize the expert container
        # We use nn.ModuleList to store M experts
        # Each expert is a nn.Sequential, simulating the LoRA logic
        self.experts = nn.ModuleList()
        self.routers = nn.ModuleList()
        self.num_experts = num_experts

        for _ in range(num_experts):
            # Other experts are randomly initialized (keep the same initialization logic as Peft)
            new_A = nn.Linear(self.in_features, r, bias=False)
            new_B = nn.Linear(r, self.out_features, bias=False)
            nn.init.kaiming_uniform_(new_A.weight, a=5 ** 0.5)
            nn.init.zeros_(new_B.weight)

            expert = nn.Sequential(new_A, new_B)
            self.experts.append(expert)

            router = nn.Linear(self.in_features, 2 * num_experts - 1)
            self.routers.append(router)

        # Save dropout and scaling, because nn.Sequential can't store these attributes
        self.scaling = scaling
        self.dropout = nn.Dropout(dropout)
        
    def register_expert(self, assigned_idx):
        self.assigned_idx = assigned_idx
        self.register_buffer('expert_map', self._build_expert_map())
        for idx, expert in enumerate(self.experts):
            expert.requires_grad = (idx == assigned_idx)

    def _build_expert_map(self):
        other_experts = [i for i in range(self.num_experts) if i != self.assigned_idx]
        device = next(self.base_layer.parameters()).device
        return torch.tensor([self.assigned_idx] * self.num_experts + other_experts, 
                           dtype=torch.long, device=device)

    def forward(self, x):
        base_out = self.base_layer(x)

        router = self.routers[self.assigned_idx]
        target_dtype = router.weight.dtype
        x = x.to(target_dtype)

        # --- Router logic (keep the same) ---
        router_logits = router(x)
        router_probs = torch.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, k=self.num_experts, dim=-1)
        mapped_expert_ids = self.expert_map[top_k_indices]

        lora_out = torch.zeros_like(base_out, dtype=target_dtype)

        # --- Calculation logic ---
        for i, expert in enumerate(self.experts):
            mask = (mapped_expert_ids == i)
            if mask.any():
                weight = (top_k_probs * mask.float()).sum(dim=-1).unsqueeze(-1)
                if weight.any():
                    # Note: here we manually do dropout and scaling
                    # Because our expert is now just a pure Linear+Linear
                    expert_val = expert(self.dropout(x))
                    lora_out += weight * expert_val * self.scaling

        return base_out + lora_out.to(dtype=base_out.dtype)


class Client(FTBaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.cluster_id = -1

    @time_record
    def run(self, model):
        for m in model.modules():
            if isinstance(m, FedLEASEMoELayer): m.register_expert(self.cluster_id)
        super().run(model)
        self.lora = {k: v.clone() for k, v in model.state_dict().items()
                     if (f'experts.{self.cluster_id}' in k or f'routers.{self.cluster_id}' in k)}

    def probe_train(self, model):
        # super().run(model)
        self.lora = {k: v.clone() for k, v in model.state_dict().items() if "lora_" in k}

    def get_lora_b_flat(self):
        b_params = [self.lora[k].detach().view(-1) for k in self.lora.keys() if "lora_B" in k]
        return b_params

class Server(FTBaseServer):
    def __init__(self, args, clients):
        super().__init__(args, clients)
        for client in self.clients:
            client.probe_train(self.model)
            self.model.load_state_dict(self.global_lora, strict=False)

        self.device = next(self.model.parameters()).device
        self.global_lora = {k: v.clone() for k, v in self.model.state_dict().items()
                            if "experts" in k or 'routers' in k}
        self.num_experts = 0
        self.experts = {}
        self.init_cluster_and_experts()
        self.init_moe_model()



    def run(self):
        self.sample()
        self.local_run()
        self.aggregate()

    def aggregate(self):
        for cluster_id in range(self.num_experts):
            clients_in_cluster = [client for client in self.clients if client.cluster_id == cluster_id]
            data_sum = sum([len(client.dataset['train']) for client in clients_in_cluster])
            from collections import defaultdict
            aggregated = defaultdict(lambda: 0)

            for client in clients_in_cluster:
                model = client.lora
                for k, v in model.items():
                    aggregated[k] = aggregated[k] + v * len(client.dataset['train']) / data_sum

            for k, v in aggregated.items(): self.global_lora[k] = v

        self.model.load_state_dict(self.global_lora, strict=False)


    def init_moe_model(self):
        def fresh_model(model):
            for name, module in model.named_children():
                if isinstance(module, LoraLayer):
                    new_layer = FedLEASEMoELayer(peft_layer=module,
                                                 num_experts=self.num_experts)
                    new_layer = new_layer.to(self.device)
                    setattr(model, name, new_layer)
                else:
                    fresh_model(module)
        fresh_model(self.model)
        self.model.load_state_dict(self.experts, strict=False)

    def init_cluster_and_experts(self):
        lora_b_list = [client.get_lora_b_flat() for client in self.clients]

        n = len(self.clients)
        d = torch.zeros(n, n)

        L = len(lora_b_list[0])
        for i in range(n):
            for j in range(n):
                dist = 0.0
                for l in range(L):
                    Bi_l = lora_b_list[i][l]
                    Bj_l = lora_b_list[j][l]
                    cos_sim = torch.dot(Bi_l, Bj_l) / (torch.norm(Bi_l) * torch.norm(Bj_l) + 1e-8)
                    dist += (1 - cos_sim)
                d[i, j] = dist / L

        dist_matrix_np = d.cpu().numpy()
        client_labels, num_experts = get_optimal_clusters(dist_matrix_np, max_experts=8)
        self.num_experts = num_experts

        all_client_params = [c.lora for c in self.clients]
        for cluster_id in range(num_experts):
            cluster_indices = [i for i, label in enumerate(client_labels) if label == cluster_id]

            # assign cluster id
            for i in cluster_indices: self.clients[i].cluster_id = cluster_id

            if not cluster_indices: continue

            print(f"Initializing Expert {cluster_id} with clients: {cluster_indices}")

            from collections import defaultdict
            avg_params = defaultdict(lambda: 0)

            for i in cluster_indices:
                params = all_client_params[i]
                for k in params.keys():
                    avg_params[k] += params[k] / len(cluster_indices)
            for k, v in avg_params.items():
                new_k = k.replace('lora_A.default.weight', f'experts.{cluster_id}.0.weight')
                new_k = k.replace('lora_B.default.weight', f'experts.{cluster_id}.1.weight')
                self.experts[new_k] = v

def get_optimal_clusters(distance_matrix, max_experts=8):
    n_clients = distance_matrix.shape[0]

    if n_clients <= 2:
        return np.zeros(n_clients, dtype=int), 1

    best_score = -2.0
    best_k = 2
    best_labels = np.zeros(n_clients, dtype=int)

    search_range = range(2, min(n_clients, max_experts + 1))

    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    for k in search_range:
        clustering = AgglomerativeClustering(
            n_clusters=k,
            metric='precomputed',
            linkage='average'
        )
        labels = clustering.fit_predict(distance_matrix)

        score = silhouette_score(distance_matrix, labels, metric='precomputed')

        print(f"K={k}, Silhouette Score={score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    print(f"Selected Optimal Experts: {best_k} (Score: {best_score:.4f})")
    return best_labels, best_k