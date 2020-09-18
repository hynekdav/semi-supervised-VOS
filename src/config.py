# -*- encoding: utf-8 -*-
# ! python3


from __future__ import annotations
from __future__ import generator_stop

import torch

class Config(object):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SCALE = 0.125
