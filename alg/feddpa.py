import math
import random

import torch
import torch.nn.functional as F

from transformers import Trainer
from alg.ftbase import FTBaseClient, FTBaseServer
from utils.time_utils import time_record

def add_args(parser):
    parser.add_argument('--alpha', type=float, default=0.5, help="Global-local Merge Ratio")
    parser.add_argument('--lambda_scale', type=float, default=0.1, help="Lambda Scale for Alpha Calculation")
    return parser.parse_args()

class Client(FTBaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.local_lora = None
        self.alpha = args.alpha
        
    @time_record
    def run(self, model):
        print(f'\nClient {self.id} starting...')
        client_model = model
        client_model.train()
        
        global_lora = {k: v.clone() for k, v in model.state_dict().items() if "lora_" in k}

        Trainer(
            model=client_model,
            args=self.training_args,
            train_dataset=self.dataset['train'],
            processing_class=self.tokenizer,
        ).train()

        self.lora = {k: v.clone() for k, v in client_model.state_dict().items() if "lora_" in k}
        if self.local_lora is None:
            self.local_lora = {k: v.clone() for k, v in client_model.state_dict().items() if "lora_" in k}

        merged_lora = {}
        for k in global_lora.keys():
            if "lora_" in k:
                merged_lora[k] = (1 - self.alpha) * global_lora[k] + self.alpha * self.local_lora[k]
            else:
                merged_lora[k] = global_lora[k]

        client_model.load_state_dict(merged_lora, strict=False)
        Trainer(
            model=client_model,
            args=self.training_args,
            train_dataset=self.dataset['train'],
            processing_class=self.tokenizer,
        ).train()
        
        self.local_lora = {k: v.clone() for k, v in client_model.state_dict().items() if "lora_" in k}

    def local_test(self, model):
        model.eval()
        total_loss = 0
        total_steps = 0
        
        test_loader = torch.utils.data.DataLoader(
            self.dataset["test"], 
            batch_size=1,
            shuffle=False
        )
        
        from tqdm import tqdm
        
        for batch in tqdm(test_loader):
            input_ids = torch.stack(batch['input_ids']).transpose(0, 1).to(model.device)
            labels = torch.stack(batch['labels']).transpose(0, 1).to(model.device)

            # --- FedDPA Key Steps ---
            # 1. Calculate the dynamic alpha_t for the current instance
            alpha_t = self.get_instance_alpha(model, input_ids)
            
            # 2. Real-time fusion and update model parameters
            merged_lora = self.apply_dual_lora(model, alpha_t)
            
            original_lora = {k: v.clone() for k, v in model.state_dict().items() if "lora_" in k}
            model.load_state_dict(merged_lora, strict=False)
            
            # 3. Forward propagation
            with torch.no_grad():
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                total_steps += 1
                
            model.load_state_dict(original_lora, strict=False)

        avg_loss = total_loss / total_steps if total_steps > 0 else 0
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")

        metrics = {
            "eval_loss": avg_loss,
            "perplexity": perplexity
        }

        print(f"Client {self.id} FedDPA metrics:", metrics)
        return metrics
    
    def get_instance_alpha(self, model, input_ids):
        # 1. 提取当前输入的特征 (使用 LLM 的隐藏状态)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_hidden_states=True)
            w_x = outputs.hidden_states[-1][:, -1, :] 

        # 2. 采样本地训练集特征 (S=5)
        S = 5
        indices = random.sample(range(len(self.dataset['train'])), min(S, len(self.dataset['train'])))
        
        sim_list = []
        for idx in indices:
            local_input = torch.tensor([self.dataset['train'][idx]['input_ids']]).to(model.device)
            with torch.no_grad():
                local_outputs = model(input_ids=local_input, output_hidden_states=True)
                w_local = local_outputs.hidden_states[-1][:, -1, :]
                sim = F.cosine_similarity(w_x, w_local)
                sim_list.append(sim)
        
        avg_sim = torch.stack(sim_list).mean()
        alpha_t = self.args.lambda_scale * avg_sim 
        return alpha_t.clamp(0, 1).item()

    def apply_dual_lora(self, model, alpha_t):
        merged_dict = {}
        global_lora = {k: v.clone() for k, v in model.state_dict().items() if "lora_" in k}
        for k in global_lora.keys():
            if "lora_" in k:
                merged_dict[k] = (1 - alpha_t) * global_lora[k].to(model.device) + \
                                alpha_t * self.local_lora[k].to(model.device)
        return merged_dict
        
class Server(FTBaseServer):
    def run(self):
        self.sample()
        self.local_run()
        self.aggregate()