# -*- encoding: utf-8 -*-
# ! python3



import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SCALE = 0.125
