import math
import torch
import numpy as np
import torch.nn as nn

from collections import defaultdict
from torch.optim import AdamW
from torch.amp import GradScaler, autocast

from alg.ftbase import FTBaseClient, FTBaseServer
from utils.time_utils import time_record


def add_args(parser):
    parser.add_argument('--t_root', type=int, default=15,
                        help="Number of rounds for root stage")
    parser.add_argument('--t_cluster', type=int, default=20,
                        help="Number of rounds for cluster stage")
    parser.add_argument('--gamma_c', type=float, default=0.1,
                        help="Orthogonality penalty: root vs cluster/leaf")
    parser.add_argument('--gamma_l', type=float, default=0.1,
                        help="Orthogonality penalty: cluster vs leaf")
    parser.add_argument('--ema_lambda', type=float, default=0.9,
                        help="EMA decay factor for subspace smoothing")
    parser.add_argument('--k_min', type=int, default=2,
                        help="Minimum number of clusters")
    parser.add_argument('--k_max', type=int, default=10,
                        help="Maximum number of clusters")
    return parser.parse_args()


# ==================== Helper Functions ====================

def get_lora_scaling(model):
    """Get LoRA scaling factor (alpha/r) from PEFT model."""
    for module in model.modules():
        if hasattr(module, 'scaling') and isinstance(module.scaling, dict) and module.scaling:
            return list(module.scaling.values())[0]
    return 1.0


def prepare_model_for_stage(model, frozen_lora_dicts, active_lora=None):
    """
    Merge frozen LoRA tiers into base weights and set the active tier's LoRA.

    - frozen_lora_dicts: list of lora state dicts to merge into base weights
    - active_lora: state dict for the tier being trained; if None, zeros out
      lora_B (keeps lora_A for proper gradient flow)

    Returns a backup dict of the original base weights for later restoration.
    """
    state = {k: v.clone() for k, v in model.state_dict().items()}
    backup = {}
    scaling = get_lora_scaling(model)

    for frozen_lora in frozen_lora_dicts:
        for a_key in [k for k in frozen_lora if 'lora_A' in k]:
            b_key = a_key.replace('lora_A', 'lora_B')
            base_key = a_key.replace('.lora_A.default.weight', '.base_layer.weight')
            if b_key in frozen_lora and base_key in state:
                if base_key not in backup:
                    backup[base_key] = state[base_key].clone()
                B = frozen_lora[b_key].float().to(state[base_key].device)
                A = frozen_lora[a_key].float().to(state[base_key].device)
                delta = (B @ A * scaling).to(state[base_key].dtype)
                state[base_key] = state[base_key] + delta

    if active_lora is not None:
        for k, v in active_lora.items():
            if k in state:
                state[k] = v.to(state[k].device).to(state[k].dtype)
    else:
        # Standard LoRA init: zero lora_B, keep lora_A for gradient flow
        for k in state:
            if 'lora_B' in k:
                state[k] = torch.zeros_like(state[k])

    model.load_state_dict(state, strict=False)
    return backup


def restore_base_weights(model, backup):
    """Restore base weights to pre-merge state."""
    if not backup:
        return
    state = {k: v.clone() for k, v in model.state_dict().items()}
    for k, v in backup.items():
        state[k] = v
    model.load_state_dict(state, strict=False)


def aggregate_product_space(clients, r):
    """
    Aggregate LoRA updates via product-space SVD (FLoRA/FlexLoRA style).
    Avoids cross-terms from averaging B and A separately.
    """
    if not clients:
        return {}
    data_sum = sum(len(c.dataset['train']) for c in clients)
    lora_a_keys = [k for k in clients[0].lora if 'lora_A' in k]

    result = {}
    for a_key in lora_a_keys:
        b_key = a_key.replace('lora_A', 'lora_B')
        delta_W = sum(
            (len(c.dataset['train']) / data_sum) * c.lora[b_key].float() @ c.lora[a_key].float()
            for c in clients
        )
        U, S, Vt = torch.linalg.svd(delta_W, full_matrices=False)
        result[b_key] = U[:, :r]
        result[a_key] = torch.diag(S[:r]) @ Vt[:r, :]

    return result


# ==================== Subspace Clustering ====================

def extract_subspace(b_dict, r):
    """Compute top-r left singular vectors of each B matrix."""
    result = {}
    for key, B in b_dict.items():
        if 'lora_B' not in key:
            continue
        B_f = B.float()
        if B_f.norm() < 1e-8:
            continue
        U, _, _ = torch.linalg.svd(B_f, full_matrices=False)
        result[key] = U[:, :r]
    return result


def subspace_pairwise_distances(subspace_list, r):
    """
    Compute N×N distance matrix using principal-angle distance:
        d_ij = 1 - (1/r) * ||U_i^T U_j||²_F
    averaged over all LoRA layers.
    """
    n = len(subspace_list)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            total, count = 0.0, 0
            for key in subspace_list[i]:
                if key not in subspace_list[j]:
                    continue
                U_i = subspace_list[i][key].float()
                U_j = subspace_list[j][key].float()
                M = U_i.t() @ U_j
                cos2 = (M * M).sum() / r
                total += (1.0 - cos2.item())
                count += 1
            D[i, j] = D[j, i] = total / count if count > 0 else 1.0
    return D


def run_spectral_clustering(D, k_min, k_max):
    """
    Select K via eigengap of the normalised Laplacian, then run
    spectral clustering on the Gaussian-kernel affinity matrix.
    """
    from scipy.sparse.csgraph import laplacian
    from sklearn.cluster import SpectralClustering

    n = D.shape[0]
    off_diag = D[D > 0]
    sigma = float(np.median(off_diag)) if len(off_diag) > 0 else 1.0
    S = np.exp(-D ** 2 / (2 * sigma ** 2))
    np.fill_diagonal(S, 1.0)

    L = laplacian(S, normed=True)
    eigvals = np.sort(np.real(np.linalg.eigvals(L)))

    best_k, best_gap = k_min, -1.0
    for k in range(k_min, min(k_max + 1, n)):
        if k < len(eigvals):
            gap = eigvals[k] - eigvals[k - 1]
            if gap > best_gap:
                best_gap, best_k = gap, k

    sc = SpectralClustering(n_clusters=best_k, affinity='precomputed', random_state=42)
    labels = sc.fit_predict(S)
    print(f"[HiLoRA] Spectral clustering → K*={best_k}, eigengap={best_gap:.4f}")
    return labels, best_k


# ==================== HiLoRA Client ====================

class Client(FTBaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.cluster_id = -1
        self.leaf_lora = {}        # personal leaf LoRA (not aggregated)
        self.ema_basis = {}        # EMA-smoothed B matrices for clustering

    def _update_ema_basis(self, model):
        """Maintain EMA of normalised B matrices for subspace similarity."""
        lam = self.args.ema_lambda
        for k, v in model.state_dict().items():
            if 'lora_B' not in k:
                continue
            b = v.detach().float()
            b_hat = b / (b.norm(p='fro') + 1e-8)
            if k not in self.ema_basis:
                self.ema_basis[k] = b_hat
            else:
                ema = lam * self.ema_basis[k] + (1 - lam) * b_hat
                self.ema_basis[k] = ema / (ema.norm(p='fro') + 1e-8)

    @time_record
    def run(self, model):
        stage = self.server.stage
        print(f'\n[HiLoRA] Client {self.id} | Stage: {stage} | Cluster: {self.cluster_id}')
        if stage == 'root':
            self._run_root(model)
        elif stage == 'cluster':
            self._run_cluster(model)
        elif stage == 'leaf':
            self._run_leaf(model)

    def _run_root(self, model):
        """Root stage: standard full-LoRA local training."""
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True

        self.trainer.train(model)
        self.lora = {k: v.clone() for k, v in model.state_dict().items() if 'lora_' in k}
        self._update_ema_basis(model)

    def _run_cluster(self, model):
        """
        Cluster stage: train cluster LoRA (root already merged into base).
        Penalty: γ_c * ||B_r^T B_c||²_F
        """
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True

        root_b = {k: self.server.root_lora[k].to(model.device)
                  for k in self.server.root_lora if 'lora_B' in k}
        self._train_with_ortho(model,
                               frozen_b_dicts=[root_b],
                               gammas=[self.args.gamma_c])
        self.lora = {k: v.clone() for k, v in model.state_dict().items() if 'lora_' in k}

    def _run_leaf(self, model):
        """
        Leaf stage: train leaf LoRA (root+cluster merged into base).
        Penalty: γ_c * ||B_r^T B_l||²_F + γ_l * ||B_c^T B_l||²_F
        """
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True

        root_b = {k: self.server.root_lora[k].to(model.device)
                  for k in self.server.root_lora if 'lora_B' in k}
        cluster_b = {k: self.server.cluster_loras[self.cluster_id][k].to(model.device)
                     for k in self.server.cluster_loras[self.cluster_id] if 'lora_B' in k}
        self._train_with_ortho(model,
                               frozen_b_dicts=[root_b, cluster_b],
                               gammas=[self.args.gamma_c, self.args.gamma_l])
        self.leaf_lora = {k: v.clone() for k, v in model.state_dict().items() if 'lora_' in k}
        self.lora = self.leaf_lora

    def _train_with_ortho(self, model, frozen_b_dicts, gammas):
        """
        Custom training loop that adds cross-tier orthogonality regularisation:
            loss_total = task_loss + Σ γ_h * ||B_frozen_h^T B_active||²_F
        """
        model.train()
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.args.lr * (0.99 ** self.server.round)
        )
        scaler = GradScaler()
        accumulation_steps = self.args.grad_accum
        global_step = 0
        optimizer.zero_grad()

        for epoch in range(self.args.epoch):
            for step, batch in enumerate(self.trainer.train_loader):
                if self.args.task_type == 'SEQ_CLS':
                    input_ids = torch.stack(batch['input_ids']).transpose(0, 1).to(model.device)
                    attention_mask = torch.stack(batch['attention_mask']).transpose(0, 1).to(model.device)
                    labels = torch.tensor(batch['labels']).to(model.device)
                    with autocast('cuda'):
                        outputs = model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        labels=labels)
                        task_loss = outputs.loss
                elif self.args.task_type == 'CAUSAL_LM':
                    input_ids = torch.stack(batch['input_ids']).transpose(0, 1).to(model.device)
                    labels = torch.stack(batch['labels']).transpose(0, 1).to(model.device)
                    with autocast('cuda'):
                        outputs = model(input_ids=input_ids, labels=labels)
                        task_loss = outputs.loss
                else:
                    raise ValueError(f"Unknown task_type: {self.args.task_type}")

                # Cross-tier orthogonality regularisation
                ortho_loss = torch.tensor(0.0, device=model.device)
                for gamma, frozen_b in zip(gammas, frozen_b_dicts):
                    if gamma <= 0:
                        continue
                    for name, param in model.named_parameters():
                        if 'lora_B' in name and param.requires_grad and name in frozen_b:
                            B_frz = frozen_b[name].float()
                            M = B_frz.t() @ param.float()   # (r, r)
                            ortho_loss = ortho_loss + gamma * (M * M).sum()

                total_loss = (task_loss.float() + ortho_loss) / accumulation_steps
                scaler.scale(total_loss).backward()

                if (step + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    global_step += 1

                if global_step >= 10:
                    break

                if step % (5 * accumulation_steps) == 0:
                    print(f"Round {self.server.round} | Client {self.id} | "
                          f"Epoch {epoch + 1} | Step {step} | "
                          f"TaskLoss: {task_loss.item():.4f} | "
                          f"OrthoLoss: {ortho_loss.item():.6f}")

        if (step + 1) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    def local_test(self, model):
        return self.trainer.eval(model)


# ==================== HiLoRA Server ====================

class Server(FTBaseServer):
    def __init__(self, args, clients):
        super().__init__(args, clients)
        self.r = args.lora_rank
        self.t_root = args.t_root
        self.t_cluster = args.t_cluster

        self.stage = 'root'
        self.stage_round = 0

        # Root LoRA: globally shared, aggregated via product-space SVD
        self.root_lora = {k: v.clone() for k, v in self.model.state_dict().items()
                          if 'lora_' in k}
        # Cluster LoRAs: {cluster_id → lora_dict}, one per discovered cluster
        self.cluster_loras = {}
        self.num_clusters = 0

    # ── Main entry point ──────────────────────────────────────────────

    def run(self):
        self._check_stage_transition()
        self.sample()
        self.local_run()
        self.aggregate()
        self.stage_round += 1

    def _check_stage_transition(self):
        if self.stage == 'root' and self.stage_round >= self.t_root:
            print(f"[HiLoRA] Root stage complete ({self.t_root} rounds). "
                  f"Running LoRA-Subspace Adaptive Clustering…")
            self._do_clustering()
            self.stage = 'cluster'
            self.stage_round = 0

        elif self.stage == 'cluster' and self.stage_round >= self.t_cluster:
            print(f"[HiLoRA] Cluster stage complete ({self.t_cluster} rounds). "
                  f"Starting leaf stage…")
            self.stage = 'leaf'
            self.stage_round = 0

    # ── Local training ────────────────────────────────────────────────

    def local_run(self):
        if self.stage == 'root':
            self._root_local_run()
        elif self.stage == 'cluster':
            self._cluster_local_run()
        elif self.stage == 'leaf':
            self._leaf_local_run()

    def _root_local_run(self):
        """Standard synchronous FL training on the root LoRA."""
        self.model.load_state_dict(self.root_lora, strict=False)
        for client in self.sampled_clients:
            client.run(self.model)
            self.model.load_state_dict(self.root_lora, strict=False)
        self.wall_clock_time += max(c.training_time for c in self.sampled_clients)

    def _cluster_local_run(self):
        """
        For each client: merge root into base weights, initialise with
        cluster LoRA (or B=0 for first round), train cluster tier.
        """
        for client in self.sampled_clients:
            cluster_id = client.cluster_id
            # None → prepare_model_for_stage will zero B, keep A (proper init)
            cluster_lora = self.cluster_loras.get(cluster_id)
            backup = prepare_model_for_stage(self.model, [self.root_lora], cluster_lora)
            client.run(self.model)
            restore_base_weights(self.model, backup)
        self.wall_clock_time += max(c.training_time for c in self.sampled_clients)

    def _leaf_local_run(self):
        """
        For each client: merge root+cluster into base, load personal leaf
        LoRA (or B=0 on first appearance), train leaf tier.
        """
        for client in self.sampled_clients:
            cluster_id = client.cluster_id
            cluster_lora = self.cluster_loras.get(cluster_id, {})
            frozen = [self.root_lora, cluster_lora] if cluster_lora else [self.root_lora]
            active_lora = client.leaf_lora if client.leaf_lora else None
            backup = prepare_model_for_stage(self.model, frozen, active_lora)
            client.run(self.model)
            restore_base_weights(self.model, backup)
        self.wall_clock_time += max(c.training_time for c in self.sampled_clients)

    # ── Aggregation ───────────────────────────────────────────────────

    def aggregate(self):
        if self.stage == 'root':
            self._aggregate_root()
        elif self.stage == 'cluster':
            self._aggregate_cluster()
        # Leaf stage: no server-side aggregation (leaf LoRA is personal)

    def _aggregate_root(self):
        """Aggregate root LoRA using product-space SVD."""
        self.root_lora = aggregate_product_space(self.sampled_clients, self.r)
        self.global_lora = self.root_lora
        self.model.load_state_dict(self.root_lora, strict=False)
        print("[HiLoRA] Root LoRA aggregated.")

    def _aggregate_cluster(self):
        """Per-cluster product-space SVD aggregation."""
        for cluster_id in range(self.num_clusters):
            in_cluster = [c for c in self.sampled_clients if c.cluster_id == cluster_id]
            if in_cluster:
                self.cluster_loras[cluster_id] = aggregate_product_space(in_cluster, self.r)
        print("[HiLoRA] Cluster LoRAs aggregated.")

    # ── Clustering ────────────────────────────────────────────────────

    def _do_clustering(self):
        """LoRA-Subspace Adaptive Clustering (Section 3.2)."""
        subspaces = [extract_subspace(c.ema_basis, self.r) for c in self.clients]
        D = subspace_pairwise_distances(subspaces, self.r)
        labels, K = run_spectral_clustering(D, self.args.k_min, self.args.k_max)
        self.num_clusters = K
        for i, client in enumerate(self.clients):
            client.cluster_id = int(labels[i])
        print(f"[HiLoRA] Assignments: {[c.cluster_id for c in self.clients]}")

    # ── Evaluation ────────────────────────────────────────────────────

    def test_all(self):
        """
        Personalised evaluation:
        - Root stage   → root LoRA only
        - Cluster stage → root + cluster LoRA
        - Leaf stage   → root + cluster + leaf LoRA
        """
        all_metrics = []
        for client in self.clients:
            print(f"[HiLoRA] Testing client {client.id} "
                  f"(stage={self.stage}, cluster={client.cluster_id})")

            if self.stage == 'root':
                self.model.load_state_dict(self.root_lora, strict=False)
                metrics = client.local_test(self.model)

            elif self.stage == 'cluster':
                cluster_lora = self.cluster_loras.get(client.cluster_id)
                frozen = [self.root_lora, cluster_lora] if cluster_lora else [self.root_lora]
                backup = prepare_model_for_stage(self.model, frozen)
                metrics = client.local_test(self.model)
                restore_base_weights(self.model, backup)

            else:  # leaf
                cluster_lora = self.cluster_loras.get(client.cluster_id, {})
                frozen = [self.root_lora, cluster_lora] if cluster_lora else [self.root_lora]
                leaf_lora = client.leaf_lora if client.leaf_lora else None
                backup = prepare_model_for_stage(self.model, frozen, leaf_lora)
                metrics = client.local_test(self.model)
                restore_base_weights(self.model, backup)

            all_metrics.append(metrics)

        # Restore model to a clean state
        self.model.load_state_dict(self.root_lora, strict=False)

        res_dict = {}
        for k in all_metrics[0].keys():
            res_dict[k] = sum(m[k] for m in all_metrics) / len(all_metrics)
        return res_dict
