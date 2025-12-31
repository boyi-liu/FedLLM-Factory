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
            if "lora_B" in name: param.requires_grad = True

        self.trainer.train(client_model)
        self.lora = {k: v.clone() for k, v in client_model.state_dict().items() if "lora_B" in k}

class Server(FTBaseServer):
    def __init__(self, args, clients):
        super().__init__(args, clients)
        self.global_lora = {k: v.clone() for k, v in self.model.state_dict().items() if "lora_B" in k}

    def run(self):
        self.sample()
        self.local_run()
        self.aggregate()