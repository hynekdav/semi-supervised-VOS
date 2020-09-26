# -*- encoding: utf-8 -*-
# ! python3


import torch


class Config(object):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SCALE = 0.125
    CONTINUOUS_FRAME = 4