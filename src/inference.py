# -*- encoding: utf-8 -*-
# ! python3
import os
from pathlib import Path

import click
import torch
import torch.nn.functional as F
import torch.utils.data
from loguru import logger
from tqdm import tqdm

from src.config import Config
from src.model.predict import predict, prepare_first_frame
from src.model.vos_net import VOSNet
from src.utils.datasets import InferenceDataset
from src.utils.inference_utils import inference_hor_flip, inference_ver_flip, inference_single
from src.utils.utils import index_to_onehot, load_model, save_predictions


@click.command(name='inference')
@click.option('--ref_num', '-n', type=int, default=9, help='number of reference frames for inference')
@click.option('--data', '-d', type=click.Path(file_okay=False, dir_okay=True), required=True,
              help='path to inference dataset folder')
@click.option('--resume', '-r', type=click.Path(file_okay=True, dir_okay=False), required=True,
              help='path to the resumed checkpoint')
@click.option('--model', '-m', type=click.Choice(['resnet18', 'resnet50', 'resnet101']), default='resnet50',
              help='network architecture, resnet18, resnet50 or resnet101')
@click.option('--temperature', '-t', type=float, default=1.0, help='temperature parameter')
@click.option('--frame_range', type=int, default=40, help='range of frames for inference')
@click.option('--sigma_1', type=float, default=8.0,
              help='smaller sigma in the motion model for dense spatial weight')
@click.option('--sigma_2', type=float, default=21.0,
              help='smaller sigma in the motion model for dense spatial weight')
@click.option('--save', '-s', type=click.Path(file_okay=False, dir_okay=True), required=True,
              help='path to save predictions')
@click.option('--device', type=click.Choice(['cpu', 'cuda']), default='cuda', help='Device to run computing on.')
@click.option('--inference-strategy', type=click.Choice(['single', 'hor-flip', 'vert-flip', '2-scale', '3-scale']),
              default='single', help='Inference strategy.')
def inference_command(ref_num, data, resume, model, temperature, frame_range, sigma_1, sigma_2, save, device,
                      inference_strategy):
    inference_command_impl(ref_num, data, resume, model, temperature, frame_range, sigma_1, sigma_2, save, device,
                           inference_strategy)


def inference_command_impl(ref_num, data, resume, model, temperature, frame_range, sigma_1, sigma_2, save, device,
                           inference_strategy, disable=False):
    if Config.DEVICE.type != device:
        Config.DEVICE = torch.device(device)
    model = VOSNet(model=model)
    model = load_model(model, resume)

    model = model.to(Config.DEVICE)
    model.eval()

    data_dir = Path(data) / 'JPEGImages/480p'
    inference_dataset = InferenceDataset(data_dir, disable=disable, inference_strategy=inference_strategy)
    inference_loader = torch.utils.data.DataLoader(inference_dataset,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=1)

    # global pred_visualize, palette, d, feats_history_l, feats_history_r, label_history_l, label_history_r, weight_dense, weight_sparse
    annotation_dir = Path(data) / 'Annotations/480p'
    annotation_list = sorted(list(annotation_dir.glob('*')))
    last_video = annotation_list[0].name

    with torch.no_grad():
        if inference_strategy == 'single':
            inference_single(model, inference_loader, len(inference_dataset), annotation_dir, last_video, save,
                             sigma_1, sigma_2, frame_range, ref_num, temperature, disable)
        elif inference_strategy == 'hor-flip':
            inference_hor_flip(model, inference_loader, len(inference_dataset), annotation_dir, last_video, save,
                               sigma_1, sigma_2, frame_range, ref_num, temperature, disable)
        elif inference_strategy == 'ver-flip':
            inference_ver_flip(model, inference_loader, len(inference_dataset), annotation_dir, last_video, save,
                               sigma_1, sigma_2, frame_range, ref_num, temperature, disable)
        elif inference_strategy == '2-scale':
            pass
        elif inference_strategy == '3-scale':
            pass

    logger.info('Inference done.')
