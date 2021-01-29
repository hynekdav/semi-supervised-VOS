# -*- encoding: utf-8 -*-
# ! python3
import math
from pathlib import Path

import click
import numpy as np
import torch
from loguru import logger
from torch import nn
from tqdm import tqdm

from src.config import Config
from src.model.loss import CrossEntropy, FocalLoss, ContrastiveLoss, TripletLoss, TripletLossWithMiner
from src.model.triplet_miners import DefaultTripletMiner
from src.model.vos_net import VOSNet
from src.utils.datasets import TrainDataset
from src.utils.utils import color_to_class, load_model


@click.command(name='train')
@click.option('--frame_num', '-n', type=int, default=10, help='number of frames to train')
@click.option('--data', '-d', type=click.Path(file_okay=False, dir_okay=True), required=True, help='path to dataset')
@click.option('--resume', '-r', type=click.Path(dir_okay=False, file_okay=True), help='path to the resumed checkpoint')
@click.option('--save_model', '-m', type=click.Path(dir_okay=True, file_okay=False), default='./checkpoints',
              help='directory to save checkpoints')
@click.option('--epochs', type=int, default=240, help='number of epochs')
@click.option('--model', type=click.Choice(['resnet18', 'resnet50', 'resnet101']), default='resnet50',
              help='network architecture, resnet18, resnet50 or resnet101')
@click.option('--temperature', '-t', type=float, default=1.0, help='temperature parameter')
@click.option('--bs', type=int, default=16, help='batch size')
@click.option('--lr', type=float, default=0.02, help='initial learning rate')
@click.option('--wd', type=float, default=3e-4, help='weight decay')
@click.option('--cj', help='use color jitter')
@click.option('--loss', type=click.Choice(['cross_entropy', 'focal', 'contrastive', 'triplet']),
              default='cross_entropy', help='Loss function to use.')
@click.option('--freeze/--no-freeze', default=True)
def train_command(frame_num, data, resume, save_model, epochs, model, temperature, bs, lr, wd, cj, loss, freeze):
    logger.info('Training started.')

    model = VOSNet(model=model)
    model = model.to(Config.DEVICE)

    if loss == 'cross_entropy':
        criterion = CrossEntropy(temperature=temperature).to(Config.DEVICE)
    elif loss == 'focal':
        criterion = FocalLoss().to(Config.DEVICE)
    elif loss == 'contrastive':
        criterion = ContrastiveLoss(temperature=temperature).to(Config.DEVICE)
    elif loss == 'triplet':
        miner = DefaultTripletMiner()
        criterion = TripletLossWithMiner(miner, temperature=temperature).to(Config.DEVICE)
    else:
        raise RuntimeError('Invalid loss type.')

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                momentum=0.9,
                                nesterov=True,
                                weight_decay=wd)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=4e-5)
    train_dataset = TrainDataset(Path(data) / 'JPEGImages/480p',
                                 Path(data) / 'Annotations/480p',
                                 frame_num=frame_num,
                                 color_jitter=cj)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=bs,
                                               shuffle=False,
                                               pin_memory=True,
                                               num_workers=8,
                                               drop_last=True)
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
    if freeze:
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


def train(train_loader, model, criterion, optimizer, epoch, centroids, batches):
    # logger.info('Starting training epoch {}'.format(epoch))
    mean_loss = []
    for i, (img_input, annotation_input, _) in tqdm(enumerate(train_loader), desc=f'Training epoch {epoch}.',
                                                    total=batches):
        (batch_size, num_frames, num_channels, H, W) = img_input.shape
        annotation_input = annotation_input.reshape(-1, 3, H, W).to(Config.DEVICE)
        annotation_input_downsample = torch.nn.functional.interpolate(annotation_input,
                                                                      scale_factor=Config.SCALE,
                                                                      mode='bilinear',
                                                                      align_corners=False)
        H_d = annotation_input_downsample.shape[-2]
        W_d = annotation_input_downsample.shape[-1]

        annotation_input = color_to_class(annotation_input_downsample, centroids)
        annotation_input = annotation_input.reshape(batch_size, num_frames, H_d, W_d)

        img_input = img_input.reshape(-1, num_channels, H, W).to(Config.DEVICE)

        features = model(img_input)
        feature_dim = features.shape[1]
        features = features.reshape(batch_size, num_frames, feature_dim, H_d, W_d)

        ref = features[:, 0:num_frames - 1, :, :, :]
        target = features[:, -1, :, :, :]
        ref_label = annotation_input[:, 0:num_frames - 1, :, :]
        target_label = annotation_input[:, -1, :, :]

        ref_label = torch.zeros(batch_size, num_frames - 1, centroids.shape[0], H_d, W_d).to(
            Config.DEVICE).scatter_(
            2, ref_label.unsqueeze(2), 1)

        loss = criterion(ref, target, ref_label, target_label)
        mean_loss.append(loss.item())
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
    return np.array(mean_loss).mean()

    # logger.info('Finished training epoch {}'.format(epoch))
