# -*- encoding: utf-8 -*-
# ! python3


import multiprocessing

import torch


class Config(object):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SCALE = 0.125
    CONTINUOUS_FRAME = 4
    CPU_COUNT = max(multiprocessing.cpu_count() - 2, 1)
