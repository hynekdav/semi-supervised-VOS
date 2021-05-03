# -*- encoding: utf-8 -*-
# ! python3
import torch
import torch.nn as nn
from torch.nn.functional import normalize

from src.model.backbone.resnet import resnet18, resnet50, resnet101


class VOSNet(nn.Module):

    def __init__(self, model='resnet50'):

        super(VOSNet, self).__init__()
        self.model = model

        if model == 'resnet18':
            resnet = resnet18(pretrained=True)
            self.backbone = nn.Sequential(*list(resnet.children())[0:8])
        elif model == 'resnet50':
            resnet = resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(resnet.children())[0:8])
            self.adjust_dim = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn256 = nn.BatchNorm2d(256)
        elif model == 'resnet101':
            resnet = resnet101(pretrained=True)
            self.backbone = nn.Sequential(*list(resnet.children())[0:8])
            self.adjust_dim = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn256 = nn.BatchNorm2d(256)
        elif model == 'facebook':
            resnet = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_swsl')
            self.backbone = nn.Sequential(*list(resnet.children())[0:8])
            self.backbone[6][0].conv2.stride = (1, 1)
            self.backbone[6][0].downsample[0].stride = (1, 1)
            self.backbone[7][0].conv2.stride = (1, 1)
            self.backbone[7][0].downsample[0].stride = (1, 1)
            self.adjust_dim = nn.Sequential(*[nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False),
                                              nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)])
            self.bn256 = nn.BatchNorm2d(256)
        else:
            raise NotImplementedError

    def forward(self, x):

        if self.model == 'resnet18':
            x = self.backbone(x)
        elif self.model == 'resnet50' or self.model == 'resnet101' or self.model == 'facebook':
            x = self.backbone(x)
            x = self.adjust_dim(x)
            x = self.bn256(x)
            x = normalize(x, p=2)

        return x

    def freeze_feature_extraction(self):
        self.backbone.requires_grad_(False)
