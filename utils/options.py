import argparse
import importlib.util
import yaml

from dataclasses import dataclass


@dataclass
class Config:
    alg: str = 'fedit'

    ### basic setting
    suffix: str = 'default'
    device: int = 0
    dataset: str = ''
    model: str = ''

    ### FL setting
    cn: int = 10
    sr: float = 1.0 # sample rate
    rnd: int = 10 # round
    tg: int = 1

    ### local training setting
    bs: int = 2  # batch size
    grad_accum: int = 8  # gradient accumulate
    epoch: int = 5
    lr: float = 1e-5

    test_gap: int = 1

    ### async
    alpha: float = 0.5



def args_parser():
    cfg = Config()
    parser = argparse.ArgumentParser()

    for field, value in cfg.__dict__.items():
        parser.add_argument(f"--{field}", type=type(value), default=value)

    # === read args from yaml ===
    with open('config.yaml', 'r') as f:
        yaml_config = yaml.load(f.read(), Loader=yaml.Loader)
    parser.set_defaults(**yaml_config)

    # === read args from command ===
    args, _ = parser.parse_known_args()

    # === read specific args from each method
    alg_module = importlib.import_module(f'alg.{args.alg}')

    spec_args = alg_module.add_args(parser) if hasattr(alg_module, 'add_args') else args
    return spec_args