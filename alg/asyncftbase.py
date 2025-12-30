import heapq
import random
from enum import Enum

from alg.ftbase import FTBaseClient, FTBaseServer

class Status(Enum):
    IDLE = 1
    ACTIVE = 2

class AsyncFTBaseClient(FTBaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.start_round = 0
        self.status = Status.IDLE

class AsyncFTBaseServer(FTBaseServer):
    def __init__(self, args, clients):
        super().__init__(args, clients)
        self.decay = args.decay
        self.client_queue = []
        self.cur_client = None

    def run(self):
        self.sample()
        self.local_run()
        self.aggregate()
        self.update_status()

    def sample(self):
        MAX_CONCURRENCY = int(len(self.clients) * self.sample_rate)
        active_num = len([c for c in self.clients if c.status == Status.ACTIVE])
        if active_num >= MAX_CONCURRENCY:
            return

        idle_clients = [c for c in self.clients if c.status != Status.ACTIVE]
        self.sampled_clients = random.sample(idle_clients, MAX_CONCURRENCY - active_num)
        for c in self.sampled_clients: c.start_round = self.round

    def local_run(self):
        for c in filter(lambda x: x.status != Status.ACTIVE, self.sampled_clients):
            c.run(self.model)
            self.model.load_state_dict(self.global_lora, strict=False)

            heapq.heappush(self.client_queue, (self.wall_clock_time + c.training_time, c))
            c.status = Status.ACTIVE
        self.wall_clock_time, self.cur_client = heapq.heappop(self.client_queue)

    def aggregate(self):
        alpha = self.decay * self.weight_decay()
        server_lora = {k: v for k, v in self.model.state_dict().items() if "lora_" in k}
        client_lora = self.cur_client.lora

        from collections import defaultdict
        aggregated = defaultdict(lambda: 0)

        for k in server_lora.keys():
            aggregated[k] = alpha * client_lora[k] + (1 - alpha) * server_lora[k]

        self.global_lora = aggregated
        self.model.load_state_dict(self.global_lora, strict=False)  # NOTE: recover the model
        print("Aggregated model updated.")

        
    def update_status(self):
        self.cur_client.status = Status.IDLE
        
    def weight_decay(self):
        return 1