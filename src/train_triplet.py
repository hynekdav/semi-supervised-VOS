# -*- encoding: utf-8 -*-
# ! python3
import math
from pathlib import Path

import click
import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import random

from src.config import Config
from src.model.loss import MinTripletLoss
from src.model.vos_net import VOSNet
from src.utils.datasets import TripletLossTrainDataset
from src.utils.utils import color_to_class, load_model


@click.command(name='train_triplet')
@click.option('--data', '-d', type=click.Path(file_okay=False, dir_okay=True), required=True, help='path to dataset')
@click.option('--resume', '-r', type=click.Path(dir_okay=False, file_okay=True), help='path to the resumed checkpoint')
@click.option('--save_model', '-s', type=click.Path(dir_okay=True, file_okay=False), default='./checkpoints',
              help='directory to save checkpoints')
@click.option('--epochs', type=int, default=240, help='number of epochs')
@click.option('--bs', type=int, default=1, help='batch size')
@click.option('--lr', type=float, default=0.02, help='initial learning rate')
@click.option('--wd', type=float, default=3e-4, help='weight decay')
def train_triplet_command(data, resume, save_model, epochs, bs, lr, wd):
    logger.info('Training started.')

    model = VOSNet(model='resnet50')
    model = model.to(Config.DEVICE)

    criterion = MinTripletLoss()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                momentum=0.9,
                                nesterov=True,
                                weight_decay=wd)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=4e-5)

    train_dataset = TripletLossTrainDataset(Path(data) / 'JPEGImages/480p', Path(data) / 'Annotations/480p')
    #         train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=bs,
                                               shuffle=True,
                                               pin_memory=False,
                                               num_workers=4,
                                               drop_last=False)
    batches = math.ceil(len(train_dataset) / bs)

    start_epoch = 0
    if resume is not None:
        try:
            model = load_model(model, resume)
        except Exception:
            model = nn.DataParallel(model)
            model = load_model(model, resume)

    save_model = Path(save_model)
    if not save_model.exists():
        save_model.mkdir(parents=True)

    centroids = np.load("./annotation_centroids.npy")
    centroids = torch.Tensor(centroids).float().to(Config.DEVICE)

    model.train()
    model.freeze_feature_extraction()
    for epoch in tqdm(range(start_epoch, start_epoch + epochs), desc='Training.'):
        loss = train(train_loader, model, criterion, optimizer, epoch, centroids, batches)
        scheduler.step()

        checkpoint_name = 'checkpoint-epoch-{:03d}-{}.pth.tar'.format(epoch, loss)
        save_path = save_model / checkpoint_name
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, save_path)
    logger.info('Training finished.')


def preprocess(img_input, annotation_input, model, centroids):
    img_input = img_input.to(Config.DEVICE)
    annotation_input = annotation_input.to(Config.DEVICE)
    annotation_input_downsample = torch.nn.functional.interpolate(annotation_input,
                                                                  scale_factor=Config.SCALE,
                                                                  mode='bilinear',
                                                                  align_corners=False)
    labels = color_to_class(annotation_input_downsample, centroids)
    features = model(img_input)
    if features.shape[3] != labels.shape[2]:
        features = features[:, :, :, :-1]
    features = features.reshape(-1, labels.shape[-1] * labels.shape[-2]).squeeze().permute(1, 0)
    labels = labels.squeeze().reshape(labels.shape[-1] * labels.shape[-2])

    features = F.normalize(features, p=2, dim=1)

    return features, labels


def train(train_loader, model, criterion, optimizer, epoch, centroids, batches):
    mean_loss = []
    for sequence in tqdm(train_loader, desc=f'Training epoch {epoch}.', total=batches):
        for idx, (img_input, annotation_input) in tqdm(enumerate(sequence), desc='Processing sequence.',
                                                       total=len(sequence)):
            positive, negative = random.sample(sequence[:idx] + sequence[idx + 1:], k=2)

            img_input = img_input.to(Config.DEVICE)
            (_, num_channels, H, W) = img_input.shape

            features, labels = preprocess(img_input, annotation_input, model, centroids)
            positive_features, positive_labels = preprocess(*positive, model, centroids)
            negative_features, negative_labels = preprocess(*negative, model, centroids)

            loss = torch.tensor(0.0, device=Config.DEVICE, requires_grad=True)
            used_labels = 0
            for label in torch.unique(labels):
                positive_pool = positive_features[positive_labels == label]
                negative_pool = negative_features[negative_labels != label]
                if positive_pool.numel() == 0 or negative_pool.numel() == 0:
                    continue
                used_labels += 1
                loss = loss + criterion(features[labels == label], positive_pool, negative_pool)

            loss = loss / max(used_labels, 1)
            mean_loss.append(loss.item())
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()
    return np.array(mean_loss).mean()
