# -*- encoding: utf-8 -*-
# ! python3


from __future__ import annotations
from __future__ import generator_stop

import torch
import torch.nn.functional as F

from src.config import Config
from src.model.predict import prepare_first_frame, predict, to_sparse
from src.utils.utils import save_predictions, index_to_onehot
from tqdm import tqdm


def inference_single(model, inference_loader, total_len, annotation_dir, last_video, save, sigma_1, sigma_2,
                     frame_range, ref_num, temperature, disable):
    global pred_visualize, palette, feats_history, label_history, weight_dense, weight_sparse, d
    frame_idx = 0
    for input, (current_video,) in tqdm(inference_loader, total=total_len, disable=disable):
        if current_video != last_video:
            # save prediction
            pred_visualize = pred_visualize.cpu().numpy()
            save_predictions(pred_visualize, palette, save, last_video)
            frame_idx = 0
        if frame_idx == 0:
            input = input.to(Config.DEVICE)
            with torch.cuda.amp.autocast():
                feats_history = model(input)
            first_annotation = annotation_dir / current_video / '00000.png'
            label_history, d, palette, weight_dense, weight_sparse = prepare_first_frame(
                current_video,
                save,
                first_annotation,
                sigma_1,
                sigma_2,
                inference_strategy='single')
            frame_idx += 1
            last_video = current_video
            label_history = to_sparse(label_history)
            continue
        (batch_size, num_channels, H, W) = input.shape

        input = input.to(Config.DEVICE)
        with torch.cuda.amp.autocast():
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
        new_label = to_sparse(index_to_onehot(torch.argmax(prediction, 0), d).unsqueeze(1))
        label_history = torch.cat((label_history, new_label), 1)
        feats_history = torch.cat((feats_history, features), 0)

        prediction = torch.nn.functional.interpolate(prediction.view(1, d, H_d, W_d), size=(H, W), mode='nearest')
        prediction = torch.argmax(prediction, 1).cpu()  # (1, H, W)

        last_video = current_video
        frame_idx += 1

        if frame_idx == 2:
            pred_visualize = prediction
        else:
            pred_visualize = torch.cat((pred_visualize, prediction), 0)

    # save last video's prediction
    pred_visualize = pred_visualize.cpu().numpy()
    save_predictions(pred_visualize, palette, save, last_video)


def inference_hor_flip(model, inference_loader, total_len, annotation_dir, last_video, save, sigma_1, sigma_2,
                       frame_range, ref_num, temperature, disable):
    global pred_visualize, palette, feats_history_l, label_history_l, weight_dense, weight_sparse, feats_history_r, label_history_r, d
    frame_idx = 0
    for input, (current_video,) in tqdm(inference_loader, total=total_len, disable=disable):
        if current_video != last_video:
            # save prediction
            pred_visualize = pred_visualize.cpu().numpy()
            save_predictions(pred_visualize, palette, save, last_video)
            frame_idx = 0
        if frame_idx == 0:
            input_l = input[0].to(Config.DEVICE)
            input_r = input[1].to(Config.DEVICE)
            with torch.cuda.amp.autocast():
                feats_history_l = model(input_l)
                feats_history_r = model(input_r)
            first_annotation = annotation_dir / current_video / '00000.png'
            label_history_l, label_history_r, d, palette, weight_dense, weight_sparse = prepare_first_frame(
                current_video,
                save,
                first_annotation,
                sigma_1,
                sigma_2,
                inference_strategy='hor-flip')
            frame_idx += 1
            last_video = current_video
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
        prediction_r = F.interpolate(prediction_r.view(1, d, H_d, W_d), size=(H, W), mode='nearest')
        prediction_r = torch.argmax(prediction_r, 1).squeeze()  # (1, H, W)
        prediction_r = torch.fliplr(prediction_r).cpu()
        prediction_l = prediction_l.cpu()

        last_video = current_video
        frame_idx += 1

        prediction = torch.maximum(prediction_l, prediction_r).unsqueeze(0).cpu().half()

        if frame_idx == 2:
            pred_visualize = prediction
        else:
            pred_visualize = torch.cat((pred_visualize, prediction), 0)

    # save last video's prediction
    pred_visualize = pred_visualize.cpu().numpy()
    save_predictions(pred_visualize, palette, save, last_video)


def inference_ver_flip(model, inference_loader, total_len, annotation_dir, last_video, save, sigma_1, sigma_2,
                       frame_range, ref_num, temperature, disable):
    global pred_visualize, palette, feats_history_l, label_history_l, weight_dense, weight_sparse, feats_history_r, label_history_r, d
    frame_idx = 0
    for input, (current_video,) in tqdm(inference_loader, total=total_len, disable=disable):
        if current_video != last_video:
            # save prediction
            pred_visualize = pred_visualize.cpu().numpy()
            save_predictions(pred_visualize, palette, save, last_video)
            frame_idx = 0
        if frame_idx == 0:
            input_l = input[0].to(Config.DEVICE)
            input_r = input[1].to(Config.DEVICE)
            with torch.cuda.amp.autocast():
                feats_history_l = model(input_l)
                feats_history_r = model(input_r)
            first_annotation = annotation_dir / current_video / '00000.png'
            label_history_l, label_history_r, d, palette, weight_dense, weight_sparse = prepare_first_frame(
                current_video,
                save,
                first_annotation,
                sigma_1,
                sigma_2,
                inference_strategy='ver-flip')
            frame_idx += 1
            last_video = current_video
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
        prediction_r = F.interpolate(prediction_r.view(1, d, H_d, W_d), size=(H, W), mode='nearest')
        prediction_r = torch.argmax(prediction_r, 1).squeeze()  # (1, H, W)
        prediction_r = torch.fliplr(prediction_r).cpu()
        prediction_l = prediction_l.cpu()

        last_video = current_video
        frame_idx += 1

        prediction = torch.maximum(prediction_l, prediction_r).unsqueeze(0).cpu().half()

        if frame_idx == 2:
            pred_visualize = prediction
        else:
            pred_visualize = torch.cat((pred_visualize, prediction), 0)

    # save last video's prediction
    pred_visualize = pred_visualize.cpu().numpy()
    save_predictions(pred_visualize, palette, save, last_video)


def inference_2_scale(model, inference_loader, total_len, annotation_dir, last_video, save, sigma_1, sigma_2,
                      frame_range, ref_num, temperature, disable):
    global pred_visualize, palette, feats_history_o, label_history_o, weight_dense_o, weight_sparse_o, feats_history_u, label_history_u, weight_dense_u, weight_sparse_u
    frame_idx = 0
    for input, (current_video,) in tqdm(inference_loader, total=total_len, disable=disable):
        if current_video != last_video:
            # save prediction
            pred_visualize = pred_visualize.cpu().numpy()
            save_predictions(pred_visualize, palette, save, last_video)
            frame_idx = 0
        if frame_idx == 0:
            input_o = input[0].to(Config.DEVICE)
            input_u = input[1].to(Config.DEVICE)
            with torch.cuda.amp.autocast():
                feats_history_o = model(input_o)
                feats_history_u = model(input_u)
            first_annotation = annotation_dir / current_video / '00000.png'
            label_history, d, palette, weight_dense, weight_sparse = prepare_first_frame(
                current_video,
                save,
                first_annotation,
                sigma_1,
                sigma_2,
                inference_strategy='2-scale')
            frame_idx += 1
            last_video = current_video
            label_history_o, label_history_u = label_history
            weight_dense_o, weight_dense_u = weight_dense
            weight_sparse_o, weight_sparse_u = weight_sparse
            continue
        (_, _, H, W) = input[0].shape

        input_o = input[0].to(Config.DEVICE)
        input_u = input[1].to(Config.DEVICE)
        with torch.cuda.amp.autocast():
            features_o = model(input_o)
            features_u = model(input_u)

        (_, feature_dim, H_d, W_d) = features_o.shape
        prediction_o = predict(feats_history_o,
                               features_o[0],
                               label_history_o,
                               weight_dense_o,
                               weight_sparse_o,
                               frame_idx,
                               frame_range,
                               ref_num,
                               temperature)
        # Store all frames' features
        new_label_o = index_to_onehot(torch.argmax(prediction_o, 0), d).unsqueeze(1)
        label_history_o = torch.cat((label_history_o, new_label_o), 1)
        feats_history_o = torch.cat((feats_history_o, features_o), 0)

        prediction_o = torch.nn.functional.interpolate(prediction_o.view(1, d, H_d, W_d), size=(H, W), mode='nearest')
        prediction_o = torch.argmax(prediction_o, 1).cpu()  # (1, H, W)

        (_, feature_dim, H_d, W_d) = features_u.shape
        prediction_u = predict(feats_history_u,
                               features_u[0],
                               label_history_u,
                               weight_dense_u,
                               weight_sparse_u,
                               frame_idx,
                               frame_range,
                               ref_num,
                               temperature)
        # Store all frames' features
        new_label_u = index_to_onehot(torch.argmax(prediction_u, 0), d).unsqueeze(1)
        label_history_u = torch.cat((label_history_u, new_label_u), 1)
        feats_history_u = torch.cat((feats_history_u, features_u), 0)

        prediction_u = torch.nn.functional.interpolate(prediction_u.view(1, d, H_d, W_d), size=(H, W), mode='nearest')
        prediction_u = torch.argmax(prediction_u, 1).cpu()  # (1, H, W)

        prediction = torch.maximum(prediction_o, prediction_u).cpu().half()

        last_video = current_video
        frame_idx += 1

        if frame_idx == 2:
            pred_visualize = prediction
        else:
            pred_visualize = torch.cat((pred_visualize, prediction), 0)

    # save last video's prediction
    pred_visualize = pred_visualize.cpu().numpy()
    save_predictions(pred_visualize, palette, save, last_video)


def inference_3_scale(model, inference_loader, total_len, annotation_dir, last_video, save, sigma_1, sigma_2,
                      frame_range, ref_num, temperature, disable):
    pass
