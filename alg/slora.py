import torch

from alg.ftbase import FTBaseClient, FTBaseServer
from utils.time_utils import time_record

def add_args(parser):
    parser.add_argument("--r1", type=float, default=0.5)
    parser.add_argument("--sft_density", type=float, default=0.1)

class Client(FTBaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.phase = 1
        self.sft_mask = None

        self.state = {}

    def set_phase(self, phase, mask=None):
        self.phase = phase
        self.sft_mask = mask

    @time_record
    def run(self, model):
        if self.phase == 1:
            # Phase 1: Sparse FT
            for name, param in model.named_parameters():
                if self.sft_mask and name in self.sft_mask:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            # 使用现有 Trainer 训练
            super().run(model)
            # 获取完整模型参数中相对于初始权重的变化 delta_W
            self.update = {k: v for k, v in model.state_dict().items() if "lora_" not in k}
        else:
            # 阶段 2: 标准 LoRA 训练
            # 这部分可以直接调用父类的 run 方法，因为它已经处理了 lora_ 权重的提取
            super().run(model)

class Server(FTBaseServer):
    def __init__(self, args, clients):
        super().__init__(args, clients)
        r1 = args.r1
        self.r1_round = r1 * args.total_round
        self.sft_density = args.sft_density
        self.sft_mask = self.generate_mask()

        for client in self.clients:
            client.set_phase(1, self.sft_mask)

        self.global_state = self.model.state_dict()

    def run(self):
        self.sample()
        self.local_run()
        self.aggregate()

    def local_run(self):
        for client in self.sampled_clients:
            client.run(self.model)
            if self.round < self.r1_round:
                self.model.load_state_dict(self.global_state, strict=False)
            else:
                self.model.load_state_dict(self.global_lora, strict=False)
        self.wall_clock_time += max([c.training_time for c in self.sampled_clients])


    def generate_mask(self):
        mask = {}
        for name, param in self.model.named_parameters():
            if "weight" in name:
                m = torch.rand(param.size()) < self.sft_density
                mask[name] = m.to(param.device)
        return mask

    def perform_svd_init(self, delta_W, r):
        U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)
        A = Vh[:r, :]
        B = U[:, :r] @ torch.diag(S[:r])
        return A, B

    def aggregate(self):
        if self.round < self.r1_round:
            # Phase 1: Aggregate Masked Model
            super().aggregate()
        elif self.round == self.r1_round:
            # 关键点：从阶段 1 切换到阶段 2
            for client in self.clients:
                client.set_phase(2)
            print("Switching to Stage 2 (LoRA) with SVD initialization.")
            # ... 执行 SVD 并注入模型 ...
            self.perform_svd_init() # TODO: change here
        else:
            # Phase 2: Aggregate the LoRA only
            super().aggregate()