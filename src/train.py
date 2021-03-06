# -*- encoding: utf-8 -*-
# ! python3
import json
import math
from pathlib import Path

import click
import numpy as np
import torch
from loguru import logger
from torch import nn
import torch.nn.functional
from torch.nn import DataParallel
from tqdm import tqdm

from src.config import Config
from src.model.loss import CrossEntropy, FocalLoss, ContrastiveLoss, TripletLossWithMiner
from src.model.triplet_miners import get_miner, TemporalMiner, OneBackOneAheadMiner, SkeletonTemporalMiner
from src.model.vos_net import VOSNet
from src.utils.datasets import TrainDataset
from src.utils.early_stopping import EarlyStopping
from src.utils.utils import color_to_class, load_model


@click.command(name='train')
@click.option('--frame_num', '-n', type=int, default=10, help='Number of frames to train.')
@click.option('--train', '-t', 'training', type=click.Path(file_okay=False, dir_okay=True), required=True,
              help='Path to training dataset.')
@click.option('--val', '-v', 'validation', type=click.Path(file_okay=False, dir_okay=True), required=True,
              help='Path to validation dataset.')
@click.option('--resume', '-r', type=click.Path(dir_okay=False, file_okay=True), help='Path to the trained checkpoint.')
@click.option('--save_model', '-m', type=click.Path(dir_okay=True, file_okay=False), default='./checkpoints',
              help='Directory to save checkpoints.')
@click.option('--epochs', type=int, default=240, help='Number of epochs.')
@click.option('--bs', type=int, default=16, help='Batch size.')
@click.option('--lr', type=float, default=0.02, help='Initial learning rate.')
@click.option('--loss', type=click.Choice(['cross_entropy', 'focal', 'contrastive', 'triplet']),
              default='cross_entropy', help='Loss function to use.')
@click.option('--freeze/--no-freeze', default=False, help='Whether or not freeze the feature extractor backbone.')
@click.option('--miner', type=click.Choice(['default', 'kernel_7x7', 'temporal', 'one_back_one_ahead',
                                            'euclidean', 'manhattan', 'chebyshev', 'skeleton',
                                            'skeleton_nearest_negative', 'skeleton_temporal']),
              default='default', help='Triplet loss miner.')
@click.option('--margin', type=click.FloatRange(min=0.0, max=1.0), default=0.1, help='Triplet loss margin.')
@click.option('--loss_weight', type=click.FloatRange(min=0.0), default=1.0, help='Weight of triplet loss.')
@click.option('--max_triplets', type=click.IntRange(min=0), default=0,
              help='Maximum number of triplets to mine (0 means take all triplets).')
@click.option('--early-stop/--no-early-stop', default=False, help='Whether or not use the early stopping for training.')
def train_command(frame_num, training, validation, resume, save_model, epochs, bs, lr, loss, freeze, miner, margin,
                  loss_weight, max_triplets, early_stop):
    logger.info('Training started.')
    torch.manual_seed(42)

    temperature = 1.0
    model = 'resnet50'
    model = VOSNet(model=model)
    model = model.to(Config.DEVICE)

    if loss == 'cross_entropy':
        criterion = CrossEntropy(temperature=temperature).to(Config.DEVICE)
    elif loss == 'focal':
        criterion = FocalLoss().to(Config.DEVICE)
    elif loss == 'contrastive':
        criterion = ContrastiveLoss(temperature=temperature).to(Config.DEVICE)
    elif loss == 'triplet':
        miner = get_miner(miner)
        miner.max_triplets = max_triplets
        if miner is None:
            raise RuntimeError('Invalid miner type.')
        criterion = TripletLossWithMiner(miner, margin=margin, temperature=temperature, weights=(1.0, loss_weight)) \
            .to(Config.DEVICE)
    else:
        raise RuntimeError('Invalid loss type.')

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                momentum=0.9,
                                nesterov=True,
                                weight_decay=3e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=4e-5)
    train_dataset = TrainDataset(Path(training) / 'JPEGImages/480p',
                                 Path(training) / 'Annotations/480p',
                                 frame_num=frame_num,
                                 color_jitter=False)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=bs,
                                               shuffle=False,
                                               pin_memory=True,
                                               num_workers=8,
                                               drop_last=True)

    validation_dataset = TrainDataset(Path(validation) / 'JPEGImages/480p',
                                      Path(validation) / 'Annotations/480p',
                                      frame_num=frame_num,
                                      color_jitter=False)
    validation_bs = max(bs // 2, 1)
    validation_loader = torch.utils.data.DataLoader(validation_dataset,
                                                    batch_size=validation_bs,
                                                    shuffle=False,
                                                    pin_memory=True,
                                                    num_workers=8,
                                                    drop_last=True)

    train_batches = math.ceil(len(train_dataset) / bs)
    validation_batches = math.ceil(len(train_dataset) / validation_bs)

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
        if isinstance(model, DataParallel):
            model.module.freeze_feature_extraction()
        else:
            model.freeze_feature_extraction()

    early_stopper = EarlyStopping(save_model, trace_func=logger.info, verbose=True)
    for epoch in tqdm(range(start_epoch, start_epoch + epochs), desc='Training.'):
        torch.manual_seed(42)
        train_loss = step(train_loader, model, criterion, optimizer, epoch, centroids, train_batches, mode='train')
        validation_loss = np.nan
        if epoch % 10 == 0:
            validation_loss = step(validation_loader, model, criterion, None, epoch, centroids, validation_batches,
                                   mode='val')
        scheduler.step()

        if early_stop and early_stopper(validation_loss, epoch, model):
            logger.info('Early stopping stopped the training.')
            break

        checkpoint_name = 'checkpoint-epoch-{:03d}-{:5f}-{:5f}.pth.tar'.format(epoch, train_loss, validation_loss)
        save_path = save_model / checkpoint_name
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, save_path)
    logger.info('Training finished.')


def step(loader, model, criterion, optimizer, epoch, centroids, batches, mode='train'):
    if mode == 'train':
        model = model.train()
    else:
        model = model.eval()
    mean_loss = []
    desc = "Training" if mode == 'train' else "Validating"
    for i, (img_input, annotation_input, _) in tqdm(enumerate(loader), desc=f'{desc} epoch {epoch}.',
                                                    total=batches):
        (batch_size, num_frames, num_channels, H, W) = img_input.shape
        annotation_input = annotation_input.reshape(-1, 3, H, W).to(Config.DEVICE)
        annotation_input_downsample = torch.nn.functional.interpolate(annotation_input,
                                                                      scale_factor=Config.SCALE,
                                                                      mode='nearest')
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

        extra_embeddings, extra_labels = None, None
        if hasattr(criterion, '_miner'):
            if isinstance(criterion._miner, TemporalMiner):
                extra_embeddings = features[:, -5:, :, :, :]
                extra_labels = annotation_input[:, -5:, :, :]
            elif isinstance(criterion._miner, OneBackOneAheadMiner):
                back_embedding = features[:, -5:-3, :, :, :]
                back_labels = annotation_input[:, -5:-3, :, :]
                ahead_embedding = features[:, -2:, :, :, :]
                ahead_labels = annotation_input[:, -2:, :, :]
                target_embedding = features[:, -3, :, :, :].unsqueeze(1)
                target_labels = annotation_input[:, -3, :, :].unsqueeze(1)
                extra_embeddings = torch.cat([back_embedding, ahead_embedding, target_embedding], dim=1)
                extra_labels = torch.cat([back_labels, ahead_labels, target_labels], dim=1)
            elif isinstance(criterion._miner, SkeletonTemporalMiner):
                extra_embeddings = features[:, -5:, :, :, :].permute((0, 2, 1, 3, 4))
                extra_embeddings = extra_embeddings.reshape(batch_size, feature_dim, 5 * 32, 32)
                extra_labels = annotation_input[:, -5:, :, :].permute((0, 2, 1, 3))
                extra_labels = extra_labels.reshape(batch_size, 5 * 32, 32)

        ref_label = torch.zeros(batch_size, num_frames - 1, centroids.shape[0], H_d, W_d).to(
            Config.DEVICE).scatter_(2, ref_label.unsqueeze(2), 1)

        loss = criterion(ref, target, ref_label, target_label, extra_embeddings, extra_labels, False)
        mean_loss.append(loss.item())

        if mode == 'train':
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return np.array(mean_loss).mean()
