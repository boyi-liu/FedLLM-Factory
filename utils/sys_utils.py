import random

import numpy as np
import yaml

SCALE_FACTOR = 50

# Training time of Jetson TX2, Jetson Nano, Raspberry Pi
# `Benchmark Analysis of Jetson TX2, Jetson Nano and Raspberry PI using Deep-CNN`
device_reference = [1, 1.8125, 11.625]


def system_config():
    with open('utils/sys.yaml', 'r') as f:
        sys_config = yaml.load(f.read(), Loader=yaml.Loader)
    return sys_config

def probs_to_counts(probs, total_count):
    raw_counts = np.array(probs) * total_count
    floored = np.floor(raw_counts).astype(int)
    remainder = total_count - floored.sum()

    fractional_parts = raw_counts - floored
    indices = np.argsort(-fractional_parts)

    for i in range(remainder):
        floored[indices[i]] += 1

    return floored.tolist()

def device_config(args):
    sys_config = system_config()

    client_num = args.cn
    prop = sys_config['dev']['dev_prop']
    prop = list(map(float, prop.split(' ')))
    prop = [p / sum(prop) for p in prop]

    counts = probs_to_counts(prop, client_num)
    result = [val for val, count in zip(device_reference, counts) for _ in range(count)]
    random.shuffle(result)
    return [r * SCALE_FACTOR for r in result]