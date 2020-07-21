# -*- encoding: utf-8 -*-
# ! python3


from __future__ import annotations
from __future__ import generator_stop

import os
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.config import DEVICE
from src.model.predict import predict, prepare_first_frame
from src.model.vos_net import VOSNet
from src.utils.datasets import InferenceDataset
from src.utils.utils import save_prediction, index_to_onehot


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
def inference_command(ref_num, data, resume, model, temperature, frame_range, sigma_1, sigma_2, save):
    model = VOSNet(model=model)
    model = nn.DataParallel(model)
    model = model.to(DEVICE)

    if resume is not None:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume, map_location=DEVICE)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'".format(resume))
        else:
            print("=> no checkpoint found at '{}'".format(resume))
            exit(-1)
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

    last_video = 0
    frame_idx = 0
    with torch.no_grad():
        for i, (input, curr_video, img_original) in tqdm(enumerate(inference_loader), total=len(inference_dataset)):
            if curr_video != last_video:
                # save prediction
                pred_visualize = pred_visualize.cpu().numpy()
                for f in range(1, frame_idx):
                    save_name = str(f).zfill(5)
                    video_name = annotation_list[last_video]
                    save_prediction(np.asarray(pred_visualize[f - 1], dtype=np.int32),
                                    palette, save, save_name, video_name)

                frame_idx = 0
                tqdm.write("End of video %d. Processing a new annotation...\n" % (last_video + 1))
            if frame_idx == 0:
                input = input.to(DEVICE)
                with torch.no_grad():
                    feats_history = model(input)
                label_history, d, palette, weight_dense, weight_sparse = prepare_first_frame(curr_video,
                                                                                             save,
                                                                                             annotation_dir,
                                                                                             sigma_1,
                                                                                             sigma_2)
                frame_idx += 1
                last_video = curr_video
                continue
            (batch_size, num_channels, H, W) = input.shape
            input = input.to(DEVICE)

            features = model(input)
            (_, feature_dim, H_d, W_d) = features.shape
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
            video_name = annotation_list[last_video]
            save_prediction(np.asarray(pred_visualize[f - 1], dtype=np.int32),
                            palette, save, save_name, video_name)
    print('Finished inference.')
