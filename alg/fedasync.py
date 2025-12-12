from alg.asyncftbase import AsyncFTBaseClient, AsyncFTBaseServer

def add_args(parser):
    parser.add_argument('--a', type=int, default=1)
    parser.add_argument('--b', type=int, default=4)
    parser.add_argument('--strategy', type=str, default='hinge', help='constant/poly/hinge')
    return parser.parse_args()


class Client(AsyncFTBaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)

class Server(AsyncFTBaseServer):
    def __init__(self, args, clients):
        super().__init__(args, clients)

    def weight_decay(self):
        tau = self.round - self.cur_client.start_round
        a = self.args.a
        b = self.args.b
        strategy = self.args.strategy
        if strategy == 'poly':
            return 1 / ((tau + 1) ** abs(a))
        elif strategy == 'hinge':
            return 1 / (a * (tau + b) + 1) if tau > b else 1
        else:
            return 1
