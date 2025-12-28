from alg.ftbase import FTBaseClient, FTBaseServer
from utils.time_utils import time_record

class Client(FTBaseClient):
    @time_record
    def run(self, model):
        super().run(model)
        self.lora = {k: v.clone() for k, v in model.state_dict().items() if "lora_A" in k}

class Server(FTBaseServer):
    def __init__(self, args, clients):
        super().__init__(args, clients)
        self.global_lora = {k: v.clone() for k, v in self.model.state_dict().items() if "lora_A" in k}

    def run(self):
        self.sample()
        self.local_run()
        self.aggregate()