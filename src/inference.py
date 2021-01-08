# -*- encoding: utf-8 -*-
# ! python3


import os
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from loguru import logger

from src.config import Config
from src.model.predict import predict, prepare_first_frame
from src.model.vos_net import VOSNet
from src.utils.datasets import InferenceDataset
from src.utils.utils import save_prediction, index_to_onehot, load_model


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
@click.option('--sigma_2', type=float, default=8.0,
              help='smaller sigma in the motion model for dense spatial weight')
@click.option('--save', '-s', type=click.Path(file_okay=False, dir_okay=True), required=True,
              help='path to save predictions')
@click.option('--device', type=click.Choice(['cpu', 'cuda']), default='cuda', help='Device to run computing on.')
def inference_command(ref_num, data, resume, model, temperature, frame_range, sigma_1, sigma_2, save, device):
    if Config.DEVICE.type != device:
        Config.DEVICE = torch.device(device)
    model = VOSNet(model=model)
    try:
        model = load_model(model, resume)
    except Exception:
        model = nn.DataParallel(model)
        model = load_model(model, resume)

    model = model.to(Config.DEVICE)
    model = model.half()
    model.eval()

    data_dir = Path(data) / 'JPEGImages/480p'  # os.path.join(data, '/JPEGImages/480p')
    inference_dataset = InferenceDataset(data_dir)
    inference_loader = torch.utils.data.DataLoader(inference_dataset,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=4)

    global pred_visualize, palette, d, feats_history, label_history, weight_dense, weight_sparse
    annotation_dir = Path(data) / 'Annotations/480p'
    annotation_list = sorted(os.listdir(annotation_dir))

    last_video = annotation_list[0]
    frame_idx = 0
    video_idx = 0
    with torch.no_grad():
        for input, curr_video in tqdm(inference_loader, total=len(inference_dataset)):
            curr_video = curr_video[0]
            if curr_video != last_video:
                # save prediction
                pred_visualize = pred_visualize.cpu().numpy()
                for f in range(1, frame_idx):
                    save_name = str(f).zfill(5)
                    video_name = last_video
                    save_prediction(np.asarray(pred_visualize[f - 1], dtype=np.int32),
                                    palette, save, save_name, video_name)
                    # torch.cuda.empty_cache()

                frame_idx = 0
                logger.info("End of video %d. Processing a new annotation...\n" % (video_idx + 1))
                video_idx += 1
            if frame_idx == 0:
                input = input.to(Config.DEVICE)
                with torch.cuda.amp.autocast():
                    feats_history = model(input)
                first_annotation = annotation_dir / curr_video / '00000.png'
                label_history, d, palette, weight_dense, weight_sparse = prepare_first_frame(curr_video,
                                                                                             save,
                                                                                             first_annotation,
                                                                                             sigma_1,
                                                                                             sigma_2)
                frame_idx += 1
                last_video = curr_video
                continue
            (batch_size, num_channels, H, W) = input.shape
            input = input.to(Config.DEVICE)

            with torch.cuda.amp.autocast():
                features = model(input)
            (_, feature_dim, H_d, W_d) = features.shape
            with torch.cuda.amp.autocast():
                prediction = predict(feats_history,
                                     features[0],
                                     label_history,
                                     weight_dense,
                                     weight_sparse,
                                     frame_idx,
                                     frame_range,
                                     ref_num,
                                     temperature)
            # Store all frames' features
            new_label = index_to_onehot(torch.argmax(prediction, 0), d).unsqueeze(1)
            label_history = torch.cat((label_history, new_label), 1)
            feats_history = torch.cat((feats_history, features), 0)

            last_video = curr_video
            frame_idx += 1

            # 1. upsample, 2. argmax
            prediction = torch.nn.functional.interpolate(prediction.view(1, d, H_d, W_d),
                                                         size=(H, W),
                                                         mode='bilinear',
                                                         align_corners=False)
            prediction = torch.argmax(prediction, 1)  # (1, H, W)

            if frame_idx == 2:
                pred_visualize = prediction
            else:
                pred_visualize = torch.cat((pred_visualize, prediction), 0)

        # save last video's prediction
        pred_visualize = pred_visualize.cpu().numpy()
        for f in range(1, frame_idx):
            save_name = str(f).zfill(5)
            video_name = last_video
            save_prediction(np.asarray(pred_visualize[f - 1], dtype=np.int32),
                            palette, save, save_name, video_name)
    logger.info('Inference done.')
