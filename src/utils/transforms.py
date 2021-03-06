# -*- encoding: utf-8 -*-
# ! python3


import numbers
from pathlib import Path

import torch
from torchvision import transforms
from tqdm import tqdm


def get_crop_params(img_size, output_size):
    """ input:
        - img_size : tuple of (w, h), original image size
        - output_size: desired output size, one int or tuple
        return:
        - i
        - j
        - th
        - tw
    """
    w, h = img_size
    if isinstance(output_size, numbers.Number):
        th, tw = (output_size, output_size)
    else:
        th, tw = output_size
    if w == tw and h == th:
        return 0, 0, h, w

    i = torch.randint(low=0, high=h - th, size=(1,)).item()
    j = torch.randint(low=0, high=w - tw, size=(1,)).item()
    return i, j, th, tw


def crop(img, i, j, h, w):
    """Crop the given PIL Image.
    Args:
        img (PIL Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        PIL Image: Cropped image.
    """
    return img.crop((j, i, j + w, i + h))


class FixedColorJitter(transforms.ColorJitter):
    """
        Same ColorJitter class, only fixes the transform params once instantiated.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super(FixedColorJitter, self).__init__(brightness, contrast, saturation, hue)
        self.transform = self.get_params(self.brightness, self.contrast,
                                         self.saturation, self.hue)

    def __call__(self, img):
        return self.transform(img)


def path_walk(top, topdown=False, followlinks=False):
    """
         See Python docs for os.walk, exact same behavior but it yields Path() instances instead
    """
    names = list(top.iterdir())

    dirs = (node for node in names if node.is_dir() is True)
    nondirs = (node for node in names if node.is_dir() is False)

    if topdown:
        yield top, dirs, nondirs

    for name in dirs:
        if followlinks or name.is_symlink() is False:
            for x in path_walk(name, topdown, followlinks):
                yield x

    if topdown is not True:
        yield top, dirs, nondirs


def make_dataset(dir, class_to_idx):
    images = []
    dir = Path(dir)
    for target in tqdm(sorted(class_to_idx.keys())):
        d = dir / target
        if not d.is_dir():
            continue

        for root, _, fnames in sorted(path_walk(d)):
            root = Path(root)
            for fname in sorted(fnames):
                path = root / fname
                item = (path, class_to_idx[target])
                images.append(item)
    return images


def hflip(input: torch.Tensor) -> torch.Tensor:
    r"""Horizontally flip a tensor image or a batch of tensor images. Input must
    be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        input (torch.Tensor): input tensor

    Returns:
        torch.Tensor: The horizontally flipped image tensor

    """
    w = input.shape[-1]
    return input[..., torch.arange(w - 1, -1, -1, device=input.device)]


def vflip(input: torch.Tensor) -> torch.Tensor:
    r"""Vertically flip a tensor image or a batch of tensor images. Input must
    be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        input (torch.Tensor): input tensor

    Returns:
        torch.Tensor: The vertically flipped image tensor

    """

    h = input.shape[-2]
    return input[..., torch.arange(h - 1, -1, -1, device=input.device), :]
