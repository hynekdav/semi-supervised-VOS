# -*- encoding: utf-8 -*-
# ! python3


import os
from pathlib import Path

import click
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

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
@click.option('--sigma_2', type=float, default=21.0,
              help='smaller sigma in the motion model for dense spatial weight')
@click.option('--save', '-s', type=click.Path(file_okay=False, dir_okay=True), required=True,
              help='path to save predictions')
@click.option('--device', type=click.Choice(['cpu', 'cuda']), default='cuda', help='Device to run computing on.')
def inference_command(ref_num, data, resume, model, temperature, frame_range, sigma_1, sigma_2, save, device):
    inference_command_impl(ref_num, data, resume, model, temperature, frame_range, sigma_1, sigma_2, save, device)


def inference_command_impl(ref_num, data, resume, model, temperature, frame_range, sigma_1, sigma_2, save, device,
                           disable=False):
    if Config.DEVICE.type != device:
        Config.DEVICE = torch.device(device)
    model = VOSNet(model=model)
    model = load_model(model, resume)

    model = model.to(Config.DEVICE)
    model.eval()

    data_dir = Path(data) / 'JPEGImages/480p'
    inference_dataset = InferenceDataset(data_dir, disable=disable, horizontal_flip=True)
    inference_loader = torch.utils.data.DataLoader(inference_dataset,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=4)

    global pred_visualize, palette, d, feats_history_l, feats_history_r, label_history_l, label_history_r, weight_dense, weight_sparse
    annotation_dir = Path(data) / 'Annotations/480p'
    annotation_list = sorted(os.listdir(annotation_dir))

    last_video = annotation_list[0]
    frame_idx, video_idx = 0, 0
    for input, curr_video in tqdm(inference_loader, total=len(inference_dataset), disable=disable):
        curr_video = curr_video[0]
        if curr_video != last_video:
            # save prediction
            pred_visualize = pred_visualize.cpu().numpy()
            for f in range(1, frame_idx):
                save_name = str(f).zfill(5)
                video_name = last_video
                save_prediction(np.asarray(pred_visualize[f - 1], dtype=np.int32), palette, save, save_name, video_name)

            frame_idx = 0
            # logger.info("End of video %d. Processing a new annotation...\n" % (video_idx + 1))
            video_idx += 1
        if frame_idx == 0:
            input_l = input[0].to(Config.DEVICE)
            input_r = input[1].to(Config.DEVICE)
            with torch.cuda.amp.autocast():
                feats_history_l = model(input_l)
                feats_history_r = model(input_r)
            first_annotation = annotation_dir / curr_video / '00000.png'
            label_history_l, label_history_r, d, palette, weight_dense, weight_sparse = prepare_first_frame(curr_video,
                                                                                                            save,
                                                                                                            first_annotation,
                                                                                                            sigma_1,
                                                                                                            sigma_2,
                                                                                                            flipped_labels=True)
            frame_idx += 1
            last_video = curr_video
            continue
        (batch_size, num_channels, H, W) = input[0].shape

        input_l = input[0].to(Config.DEVICE)
        input_r = input[1].to(Config.DEVICE)
        with torch.cuda.amp.autocast():
            features_l = model(input_l)
            features_r = model(input_r)

        (_, feature_dim, H_d, W_d) = features_l.shape
        prediction_l = predict(feats_history_l,
                               features_l[0],
                               label_history_l,
                               weight_dense,
                               weight_sparse,
                               frame_idx,
                               frame_range,
                               ref_num,
                               temperature)
        # Store all frames' features
        new_label_l = index_to_onehot(torch.argmax(prediction_l, 0), d).unsqueeze(1)
        label_history_l = torch.cat((label_history_l, new_label_l), 1)
        feats_history_l = torch.cat((feats_history_l, features_l), 0)

        prediction_l = torch.nn.functional.interpolate(prediction_l.view(1, d, H_d, W_d),
                                                       size=(H, W),
                                                       mode='nearest')
        prediction_l = torch.argmax(prediction_l, 1).squeeze()  # (1, H, W)

        prediction_r = predict(feats_history_r,
                               features_r[0],
                               label_history_r,
                               weight_dense,
                               weight_sparse,
                               frame_idx,
                               frame_range,
                               ref_num,
                               temperature)
        # Store all frames' features
        new_label_r = index_to_onehot(torch.argmax(prediction_r, 0), d).unsqueeze(1)
        label_history_r = torch.cat((label_history_r, new_label_r), 1)
        feats_history_r = torch.cat((feats_history_r, features_r), 0)

        # 1. upsample, 2. argmax
        prediction_r = torch.nn.functional.interpolate(prediction_r.view(1, d, H_d, W_d),
                                                       size=(H, W),
                                                       mode='nearest')
        prediction_r = torch.argmax(prediction_r, 1).squeeze()  # (1, H, W)
        prediction_r = torch.fliplr(prediction_r).cpu()
        prediction_l = prediction_l.cpu()

        last_video = curr_video
        frame_idx += 1

        # TODO merging predictions
        prediction = torch.maximum(prediction_l, prediction_r)

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
