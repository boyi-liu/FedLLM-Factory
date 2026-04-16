import importlib.util
import sys
import numpy as np
import os

from utils.options import args_parser
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class FedSim:
    def __init__(self, args):
        self.args = args
        args.suffix = f'exp/{args.suffix}'
        os.makedirs(f'./{args.suffix}', exist_ok=True)

        output_path = f'{args.suffix}/{args.alg}_{args.dataset}_{args.model}_' \
                      f'{args.cn}c_{args.epoch}E_lr{args.lr}'
        self.output = open(f'./{output_path}.txt', 'a')

        # === route to algorithm module ===
        if args.mode == 'prototype':
            alg_module = importlib.import_module(f'alg.{args.alg}')
        elif args.mode == 'realistic':
            alg_module = importlib.import_module(f'alg.{args.alg}_event')

        # === init clients & server ===
        self.clients = [alg_module.Client(idx, args) for idx in tqdm(range(args.cn), desc="Loading clients...")]
        self.server = alg_module.Server(args, self.clients)
        
        if args.mode == 'prototype':
            self.simulate()
        elif args.mode == 'realistic':
            self.simulate_ddl()

    def simulate(self):
        TEST_GAP = self.args.test_gap

        try:
            for rnd in tqdm(range(0, self.args.rnd), desc='Communication round', leave=False):
                # ===================== train =====================
                self.server.round = rnd
                self.server.run()

                # ===================== test =====================
                if (self.args.rnd - rnd <= 10) or (rnd % TEST_GAP == (TEST_GAP-1)):
                    ret_dict = self.server.test_all()
                    
                    res_text = f'\n[Round {rnd}] Test Results:\n'
                    for k, v in ret_dict.items():
                        res_text += f"{k}: {v:.4f}\n"
                    res_text += f'Wall clock time: {self.server.wall_clock_time}\n'
                    print(res_text)       
                    self.output.write(res_text)
                    self.output.flush()

        except KeyboardInterrupt:
            ...
        finally:
            pass
        
    
    def simulate_ddl(self):
        TEST_GAP = self.args.test_gap
        try:
            while self.server.early_break is False:
                self.server.run()
                
                if (self.args.rnd - self.server.round <= 10) or (self.server.round % TEST_GAP == (TEST_GAP-1)):
                    ret_dict = self.server.test_all()
                    res_text = f'[Round {self.server.round} | Wall clock time: {self.server.wall_clock_time}]\n'
                    for k, v in ret_dict.items():
                        res_text += f"{k}: {v:.4f}\n"
                    
                    print(res_text)       
                    self.output.write(res_text)
                    self.output.flush()

        except KeyboardInterrupt:
            ...
        finally:
            pass

if __name__ == '__main__':
    FedSim(args=args_parser())