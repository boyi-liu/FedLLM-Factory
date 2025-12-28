import torch
import torch.nn as nn

from alg.ftbase import FTBaseClient, FTBaseServer
from utils.model_utils import load_model
from utils.time_utils import time_record
from peft.tuners.lora import LoraLayer

def add_args(parser):
    parser.add_argument('--num_heads', type=int, default=4, help="Number of Heads")
    return parser.parse_args()

class RAVANLayer(nn.Module, LoraLayer):
    def __init__(self, base_layer, r, num_heads, **kwargs):
        super().__init__()
        # 1. Basic Configuration
        self.base_layer = base_layer
        self.r = r
        self.num_heads = num_heads
        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # 2. Initialize A and B in RAVAN
        self.ravan_B = nn.ParameterList([
            nn.Parameter(torch.randn(out_features, r), requires_grad=False)
            for _ in range(num_heads)
        ])
        self.ravan_A = nn.ParameterList([
            nn.Parameter(torch.randn(r, in_features), requires_grad=False)
            for _ in range(num_heads)
        ])

        # 3. Initialize Trainable H_i, s_i
        self.ravan_H = nn.ParameterList([
            nn.Parameter(torch.zeros(r, r), requires_grad=True)
            for _ in range(num_heads)
        ])
        self.ravan_s = nn.ParameterList([
            nn.Parameter(torch.ones(1), requires_grad=True)
            for _ in range(num_heads)
        ])

    def forward(self, x, *args, **kwargs):
        result = self.base_layer(x, *args, **kwargs)

        for i in range(self.num_heads):
            ax = x @ self.ravan_A[i].t()  # (batch, r)
            h_ax = ax @ self.ravan_H[i].t()  # (batch, r)
            b_h_ax = h_ax @ self.ravan_B[i].t()  # (batch, out_features)

            result += self.ravan_s[i] * b_h_ax

        return result

class Client(FTBaseClient):
    @time_record
    def run(self, model):
        super().run(model)
        self.lora = {k: v.clone() for k, v in model.state_dict().items() if "ravan_" in k}

class Server(FTBaseServer):
    def __init__(self, args, clients):
        super().__init__(args, clients)
        raw_model = load_model(args)

        self.inject_ravan(raw_model, r=args.lora_rank, h=args.num_heads)

        for name, param in raw_model.named_parameters():
            if "ravan_" not in name: param.requires_grad = False

        self.model = raw_model

        self.global_lora = {k: v.clone() for k, v in self.model.state_dict().items() if "ravan_" in k}


    def run(self):
        self.sample()
        self.local_run()
        self.aggregate()

    def aggregate(self):
        total_samples = sum(len(c.dataset['train']) for c in self.sampled_clients)
        from collections import defaultdict
        new_global_state = defaultdict(lambda: 0)

        for name, param in self.global_lora.items():
            if "ravan_H" in name:
                s_name = name.replace('ravan_H', 'ravan_s')
                for client in self.sampled_clients:
                    new_global_state[name] = new_global_state[name] + client.lora[name] * client.lora[s_name] * (len(client.dataset['train']) / total_samples)
            elif "ravan_s" in name:
                # The global scaling factor s is always kept as 1 
                # since its effect has already been incorporated into H during aggregation.
                new_global_state[name] = torch.ones_like(param)
            else:
                # B and A are kept frozen
                new_global_state[name] = param

        self.model.load_state_dict(new_global_state, strict=False)
        self.global_lora = new_global_state

    def inject_ravan(self, model, r, h):
        # Recursively replace linear layers with RAVANLayer
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                if any(target in name for target in ["q_proj", "v_proj"]):
                    new_layer = RAVANLayer(module, r, h)
                    setattr(model, name, new_layer)
            else:
                self.inject_ravan(module, r, h)