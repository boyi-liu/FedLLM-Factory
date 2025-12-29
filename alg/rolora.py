from transformers import Trainer

from alg.ftbase import FTBaseClient, FTBaseServer
from utils.time_utils import time_record

class Client(FTBaseClient):
    @time_record
    def run(self, model):
        client_model = model

        # freeze all parameters first
        for param in client_model.parameters():
            param.requires_grad = False

        # then unfreeze lora_B
        for name, param in client_model.named_parameters():
            if self.server.round % 2 == 1:
                if "lora_B" in name: param.requires_grad = True
            else:
                if "lora_A" in name: param.requires_grad = True

        client_model.train()

        Trainer(
            model=client_model,
            args=self.training_args,
            train_dataset=self.dataset['train'],
            processing_class=self.tokenizer,
        ).train()

        if self.server.round % 2 == 1:
            self.lora = {k: v.clone() for k, v in client_model.state_dict().items() if "lora_B" in k}
        else:
            self.lora = {k: v.clone() for k, v in client_model.state_dict().items() if "lora_A" in k}

class Server(FTBaseServer):
    def __init__(self, args, clients):
        super().__init__(args, clients)
        if self.round % 2 == 1:
            self.global_lora = {k: v.clone() for k, v in self.model.state_dict().items() if "lora_B" in k}
        else:
            self.global_lora = {k: v.clone() for k, v in self.model.state_dict().items() if "lora_A" in k}

    def run(self):
        self.sample()
        self.local_run()
        self.aggregate()