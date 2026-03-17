import torch

from alg.ftbase import FTBaseClient, FTBaseServer
from utils.time_utils import time_record


def get_lora_scaling(model):
    """Get LoRA scaling factor (alpha/r) from PEFT model."""
    for module in model.modules():
        if hasattr(module, 'scaling') and isinstance(module.scaling, dict) and module.scaling:
            return list(module.scaling.values())[0]
    return 1.0


class Client(FTBaseClient):
    @time_record
    def run(self, model):
        super().run(model)


class Server(FTBaseServer):
    def __init__(self, args, clients):
        super().__init__(args, clients)
        self.r = args.lora_rank
        # Save original pretrained base weights (θ0) — never modified after init
        self.pretrained_base = {
            k: v.clone()
            for k, v in self.model.state_dict().items()
            if 'base_layer.weight' in k
        }

    def run(self):
        self.sample()
        self.local_run()
        self.aggregate()

    def local_run(self):
        is_first_round = (self.round == 0)

        for client in self.sampled_clients:
            # Restore pretrained base weights + load current global lora
            self.model.load_state_dict(self.pretrained_base, strict=False)
            self.model.load_state_dict(self.global_lora, strict=False)

            # First round: QR-based orthonormal initialization (Section 5.1)
            if is_first_round:
                self._qr_init(self.model)

            client.run(self.model)

            # Restore base weights after client training
            # (base may have been modified by QR init)
            self.model.load_state_dict(self.pretrained_base, strict=False)
            self.model.load_state_dict(self.global_lora, strict=False)

        self.wall_clock_time += max([c.training_time for c in self.sampled_clients])

    def _qr_init(self, model):
        """QR-based orthonormal initialization (Algorithm 1, lines 2-4).

        For each LoRA layer with base weight θ0:
          Qk, Rk = QR(θ0)
          Ak = Rk[:r, :]   (lora_A)
          Bk = Qk[:, :r]   (lora_B)
          base = θ0 - scaling * Bk @ Ak   (so effective weight = θ0)
        """
        state = model.state_dict()
        scaling = get_lora_scaling(model)
        updates = {}

        for a_key in [k for k in state if 'lora_A' in k]:
            b_key = a_key.replace('lora_A', 'lora_B')
            base_key = a_key.replace('.lora_A.default.weight', '.base_layer.weight')

            if base_key not in state:
                continue

            W0 = state[base_key].float()   # θ0 ∈ R^{d×k}
            r = state[a_key].shape[0]      # lora rank

            # Reduced QR: W0 = Q @ R,  Q: (d, min(d,k)),  R: (min(d,k), k)
            Q, R = torch.linalg.qr(W0, mode='reduced')

            r_use = min(r, Q.shape[1])
            B = Q[:, :r_use]   # ∈ R^{d×r}
            A = R[:r_use, :]   # ∈ R^{r×k}

            # Adjust base so effective weight stays at θ0:
            #   base + scaling * B @ A = θ0  =>  base = θ0 - scaling * B @ A
            updates[base_key] = (W0 - scaling * B @ A).to(state[base_key].dtype)
            updates[a_key] = A.to(state[a_key].dtype)
            updates[b_key] = B.to(state[b_key].dtype)

        model.load_state_dict(updates, strict=False)

    def aggregate(self):
        """Concatenated QR aggregation (Section 5.2 / Algorithm 2, lines 7-13).

        Constructs:
          Ac = [p1*A1; p2*A2; ...]  ∈ R^{(S·r)×k}
          Bc = [B1, B2, ...]        ∈ R^{d×(S·r)}
          ∆θ = Bc @ Ac = Σ pk * Bk @ Ak
          Q, R = QR(∆θ)
          Bs = Q[:, :rs],  As = R[:rs, :]
        """
        data_total = sum([len(c.dataset['train']) for c in self.sampled_clients])
        lora_keys = [k for k in self.global_lora.keys() if 'lora_A' in k]

        new_global_lora = {}
        for a_key in lora_keys:
            b_key = a_key.replace('lora_A', 'lora_B')

            a_list, b_list = [], []
            for client in self.sampled_clients:
                pk = len(client.dataset['train']) / data_total
                a_list.append((pk * client.lora[a_key]).float())
                b_list.append(client.lora[b_key].float())

            # Ac: (S*r, k),  Bc: (d, S*r)
            Ac = torch.cat(a_list, dim=0)
            Bc = torch.cat(b_list, dim=1)

            # ∆θ = Bc @ Ac  ∈ R^{d×k}
            delta_W = torch.matmul(Bc, Ac)

            # Reduced QR decomposition of ∆θ
            Q, R = torch.linalg.qr(delta_W, mode='reduced')  # Q: (d, min(d,k)), R: (min(d,k), k)

            rs = min(self.r, Q.shape[1])
            Bs = Q[:, :rs]   # global lora_B  ∈ R^{d×rs}
            As = R[:rs, :]   # global lora_A  ∈ R^{rs×k}

            model_state = self.model.state_dict()
            new_global_lora[a_key] = As.to(model_state[a_key].dtype)
            new_global_lora[b_key] = Bs.to(model_state[b_key].dtype)

        self.global_lora = new_global_lora
        # Restore pretrained base weights and load new global lora
        self.model.load_state_dict(self.pretrained_base, strict=False)
        self.model.load_state_dict(self.global_lora, strict=False)
        print("ILoRA: Aggregated via concatenated QR decomposition.")
