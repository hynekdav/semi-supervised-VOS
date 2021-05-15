# -*- encoding: utf-8 -*-
# ! python3


from __future__ import annotations
from __future__ import generator_stop

import json
import math
from pathlib import Path

import click
import numpy as np
import torch
import torch.utils.data
from loguru import logger
from torch import nn
from tqdm import tqdm

from src.config import Config
from src.model.loss import CrossEntropy, FocalLoss, ContrastiveLoss, TripletLossWithMiner
from src.model.triplet_miners import get_miner
from src.model.vos_net import VOSNet
from src.train import step
from src.utils.datasets import TrainDataset
from src.utils.utils import load_model


@click.command(name='validation')
@click.option('--data', '-d', type=click.Path(file_okay=False, dir_okay=True), required=True, help='Path to dataset.')
@click.option('--checkpoints', '-c', type=click.Path(dir_okay=True, file_okay=False), help='Path to checkpoints.')
@click.option('--bs', type=int, default=16, help='Batch size.')
@click.option('--loss', type=click.Choice(['cross_entropy', 'focal', 'contrastive', 'triplet']),
              default='cross_entropy', help='Loss function to use.')
@click.option('--miner', type=click.Choice(['default', 'kernel_7x7', 'temporal', 'one_back_one_ahead',
                                            'euclidean', 'manhattan', 'chebyshev', 'skeleton',
                                            'skeleton_nearest_negative', 'skeleton_temporal']),
              default='default', help='Triplet loss miner.')
@click.option('--margin', type=click.FloatRange(min=0.0, max=1.0), default=0.1, help='Triplet loss margin.')
@click.option('--loss_weight', type=click.FloatRange(min=0.0), default=6.0, help='Weight of triplet loss.')
@click.option('--output', '-o', type=click.Path(dir_okay=False, file_okay=True), help='Path to output JSON.')
def validation_command(data, checkpoints, bs, loss, miner, margin, loss_weight, output):
    logger.info('Validation started.')

    temperature = 1.0

    if loss == 'cross_entropy':
        criterion = CrossEntropy(temperature=temperature).to(Config.DEVICE)
    elif loss == 'focal':
        criterion = FocalLoss().to(Config.DEVICE)
    elif loss == 'contrastive':
        criterion = ContrastiveLoss(temperature=temperature).to(Config.DEVICE)
    elif loss == 'triplet':
        miner = get_miner(miner)
        if miner is None:
            raise RuntimeError('Invalid miner type.')
        criterion = TripletLossWithMiner(miner, margin=margin, temperature=temperature, weights=(1.0, loss_weight)) \
            .to(Config.DEVICE)
    else:
        raise RuntimeError('Invalid loss type.')

    validation_dataset = TrainDataset(Path(data) / 'JPEGImages/480p',
                                      Path(data) / 'Annotations/480p',
                                      frame_num=10,
                                      color_jitter=False)

    validation_loader = torch.utils.data.DataLoader(validation_dataset,
                                                    batch_size=bs,
                                                    shuffle=False,
                                                    pin_memory=True,
                                                    num_workers=8,
                                                    drop_last=True)
    batches = math.ceil(len(validation_dataset) / bs)

    centroids = np.load("./annotation_centroids.npy")
    centroids = torch.Tensor(centroids).float().to(Config.DEVICE)

    checkpoints = list(Path(checkpoints).glob('*.pth.tar'))
    checkpoints.sort()

    losses = {}
    for checkpoint in tqdm(checkpoints, desc=f'Validating checkpoints: '):
        model = 'resnet50'
        model = VOSNet(model=model)
        model = model.to(Config.DEVICE)

        try:
            model = load_model(model, str(checkpoint.absolute()))
        except Exception:
            model = nn.DataParallel(model)
            model = load_model(model, str(checkpoint.absolute()))
        model.eval()
        loss = step(validation_loader, model, criterion, None, 0, centroids, batches, mode='val')
        losses[checkpoint.name] = loss

    with Path(output).open(mode='w') as writer:
        json.dump(losses, writer)

    logger.info('Validation finished.')
