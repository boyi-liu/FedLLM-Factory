from alg.eventbase import EventBaseClient, EventBaseServer
from utils.time_utils import time_record

class Client(EventBaseClient):
    @time_record
    def run(self, model):
        super().run(model)

class Server(EventBaseServer):
    def run(self):
        super().run()
