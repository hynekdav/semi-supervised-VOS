# -*- encoding: utf-8 -*-
# ! python3


from __future__ import annotations
from __future__ import generator_stop

from pathlib import Path

import torch
from PIL import Image

from src.config import Config


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_prediction(prediction, palette, save_path, save_name, video_name):
    img = Image.fromarray(prediction)
    img = img.convert('L')
    img.putpalette(palette)
    img = img.convert('P')
    video_path = Path(save_path) / video_name
    if not video_path.exists():
        video_path.mkdir(parents=True)
    img.save((video_path / (save_name + '.png')).absolute())


def color_to_class(img, centroids):
    """
    Change rgb image array into class index.
    :param img: (batch_size, C, H, W)
    :param centroids:
    :return: (batch_size, H, W)
    """
    (batch_size, C, H, W) = img.shape
    img = img.permute(0, 2, 3, 1).reshape(-1, C)
    class_idx = torch.argmin(torch.sqrt(torch.sum((img.unsqueeze(1) - centroids) ** 2, 2)), 1)
    class_idx = torch.reshape(class_idx, (batch_size, H, W))
    return class_idx


def index_to_onehot(idx, d):
    """ input:
        - idx: (H*W)
        return:
        - one_hot: (d, H*W)
    """
    n = idx.shape[0]
    one_hot = torch.zeros(d, n, device=Config.DEVICE).scatter_(0, idx.view(1, -1), 1)

    return one_hot
