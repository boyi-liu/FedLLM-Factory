import logging
import os
import sys


def get_logger(args) -> logging.Logger:
    """
    Build a logger that writes to both stdout and a .log file.
    Call once in FedSim.__init__ and pass the returned logger around.
    """
    output_path = (
        f'{args.suffix}/{args.alg}_{args.dataset}_{args.model}_'
        f'{args.cn}c_{args.epoch}E_lr{args.lr}'
    )
    os.makedirs(f'./{args.suffix}', exist_ok=True)
    log_file = f'./{output_path}.log'

    logger = logging.getLogger(output_path)
    logger.setLevel(logging.INFO)

    # avoid duplicate handlers if get_logger is called more than once
    if logger.handlers:
        return logger

    fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # console
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # file  (no manual flush / close needed)
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
