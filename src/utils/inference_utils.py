# -*- encoding: utf-8 -*-
# ! python3


from __future__ import annotations
from __future__ import generator_stop

import pickle
from collections import defaultdict

import joblib
import numpy as np
import torch
import torch.nn.functional as F

from src.config import Config
from src.model.predict import prepare_first_frame, predict
from src.utils.transforms import hflip
from src.utils.utils import save_predictions, index_to_onehot
from tqdm import tqdm

REDUCTIONS = {'maximum': lambda x, y: torch.maximum(x, y),
              'minimum': lambda x, y: torch.minimum(x, y),
              'mean': lambda x, y: (x + y) / 2.0}


def inference_single(model, inference_loader, total_len, annotation_dir, last_video, save, sigma_1, sigma_2,
                     frame_range, ref_num, temperature, probability_propagation, disable):
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
                inference_strategy='single',
                probability_propagation=probability_propagation)
            frame_idx += 1
            last_video = current_video
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
                             temperature,
                             probability_propagation)
        # Store all frames' features
        if probability_propagation:
            new_label = prediction.unsqueeze(1)
        else:
            new_label = index_to_onehot(torch.argmax(prediction, 0), d).unsqueeze(1)
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
                       frame_range, ref_num, temperature, probability_propagation, reduction_str, disable):
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
                inference_strategy='hor-flip',
                probability_propagation=probability_propagation)
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
                               temperature,
                               probability_propagation)
        # Store all frames' features
        if probability_propagation:
            new_label_l = prediction_l.unsqueeze(1)
        else:
            new_label_l = index_to_onehot(torch.argmax(prediction_l, 0), d).unsqueeze(1)
        label_history_l = torch.cat((label_history_l, new_label_l), 1)
        feats_history_l = torch.cat((feats_history_l, features_l), 0)

        prediction_l = torch.nn.functional.interpolate(prediction_l.view(1, d, H_d, W_d),
                                                       size=(H, W),
                                                       mode='nearest')
        if not probability_propagation:
            prediction_l = torch.argmax(prediction_l, 1).squeeze()  # (1, H, W)

        prediction_r = predict(feats_history_r,
                               features_r[0],
                               label_history_r,
                               weight_dense,
                               weight_sparse,
                               frame_idx,
                               frame_range,
                               ref_num,
                               temperature,
                               probability_propagation)
        # Store all frames' features
        if probability_propagation:
            new_label_r = prediction_r.unsqueeze(1)
        else:
            new_label_r = index_to_onehot(torch.argmax(prediction_r, 0), d).unsqueeze(1)
        label_history_r = torch.cat((label_history_r, new_label_r), 1)
        feats_history_r = torch.cat((feats_history_r, features_r), 0)

        # 1. upsample, 2. argmax
        prediction_r = F.interpolate(prediction_r.view(1, d, H_d, W_d), size=(H, W), mode='nearest')
        if not probability_propagation:
            prediction_r = torch.argmax(prediction_r, 1).squeeze()  # (1, H, W)
        prediction_r = torch.fliplr(prediction_r).cpu()
        prediction_l = prediction_l.cpu()

        last_video = current_video
        frame_idx += 1

        if probability_propagation:
            reduction = REDUCTIONS.get(reduction_str)
            prediction = reduction(prediction_l, prediction_r).cpu().half()
            prediction = torch.argmax(prediction, 1).cpu()  # (1, H, W)
        else:
            prediction = torch.maximum(prediction_l, prediction_r).unsqueeze(0).cpu().half()

        if frame_idx == 2:
            pred_visualize = prediction
        else:
            pred_visualize = torch.cat((pred_visualize, prediction), 0)

    # save last video's prediction
    pred_visualize = pred_visualize.cpu().numpy()
    save_predictions(pred_visualize, palette, save, last_video)


def inference_ver_flip(model, inference_loader, total_len, annotation_dir, last_video, save, sigma_1, sigma_2,
                       frame_range, ref_num, temperature, probability_propagation, reduction_str, disable):
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
                inference_strategy='ver-flip',
                probability_propagation=probability_propagation)
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
                               temperature,
                               probability_propagation)
        # Store all frames' features
        if probability_propagation:
            new_label_l = prediction_l.unsqueeze(1)
        else:
            new_label_l = index_to_onehot(torch.argmax(prediction_l, 0), d).unsqueeze(1)
        label_history_l = torch.cat((label_history_l, new_label_l), 1)
        feats_history_l = torch.cat((feats_history_l, features_l), 0)

        prediction_l = torch.nn.functional.interpolate(prediction_l.view(1, d, H_d, W_d),
                                                       size=(H, W),
                                                       mode='nearest')
        if not probability_propagation:
            prediction_l = torch.argmax(prediction_l, 1).squeeze()  # (1, H, W)

        prediction_r = predict(feats_history_r,
                               features_r[0],
                               label_history_r,
                               weight_dense,
                               weight_sparse,
                               frame_idx,
                               frame_range,
                               ref_num,
                               temperature,
                               probability_propagation)
        # Store all frames' features
        if probability_propagation:
            new_label_r = prediction_r.unsqueeze(1)
        else:
            new_label_r = index_to_onehot(torch.argmax(prediction_r, 0), d).unsqueeze(1)
        label_history_r = torch.cat((label_history_r, new_label_r), 1)
        feats_history_r = torch.cat((feats_history_r, features_r), 0)

        # 1. upsample, 2. argmax
        prediction_r = F.interpolate(prediction_r.view(1, d, H_d, W_d), size=(H, W), mode='nearest')
        if not probability_propagation:
            prediction_r = torch.argmax(prediction_r, 1).squeeze()  # (1, H, W)
        prediction_r = torch.fliplr(prediction_r).cpu()
        prediction_l = prediction_l.cpu()

        last_video = current_video
        frame_idx += 1

        if probability_propagation:
            reduction = REDUCTIONS.get(reduction_str)
            prediction = reduction(prediction_l, prediction_r).cpu().half()
            prediction = torch.argmax(prediction, 1).cpu()  # (1, H, W)
        else:
            prediction = torch.maximum(prediction_l, prediction_r).unsqueeze(0).cpu().half()

        if frame_idx == 2:
            pred_visualize = prediction
        else:
            pred_visualize = torch.cat((pred_visualize, prediction), 0)

    # save last video's prediction
    pred_visualize = pred_visualize.cpu().numpy()
    save_predictions(pred_visualize, palette, save, last_video)


def inference_2_scale(model, inference_loader, total_len, annotation_dir, last_video, save, sigma_1, sigma_2,
                      frame_range, ref_num, temperature, probability_propagation, scale, reduction_str, flip_pred,
                      disable):
    global pred_visualize, palette, feats_history_o, label_history_o, weight_dense_o, weight_sparse_o, feats_history_u, label_history_u, weight_dense_u, weight_sparse_u, d
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
                inference_strategy='2-scale',
                probability_propagation=probability_propagation,
                scale=scale)
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
                               temperature,
                               probability_propagation)
        # Store all frames' features
        if probability_propagation:
            new_label_o = prediction_o.unsqueeze(1)
        else:
            new_label_o = index_to_onehot(torch.argmax(prediction_o, 0), d).unsqueeze(1)
        label_history_o = torch.cat((label_history_o, new_label_o), 1)
        feats_history_o = torch.cat((feats_history_o, features_o), 0)

        prediction_o = torch.nn.functional.interpolate(prediction_o.view(1, d, H_d, W_d), size=(H, W), mode='nearest')
        if not probability_propagation:
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
                               temperature,
                               probability_propagation)
        # Store all frames' features
        if probability_propagation:
            new_label_u = prediction_u.unsqueeze(1)
        else:
            new_label_u = index_to_onehot(torch.argmax(prediction_u, 0), d).unsqueeze(1)
        label_history_u = torch.cat((label_history_u, new_label_u), 1)
        feats_history_u = torch.cat((feats_history_u, features_u), 0)

        prediction_u = torch.nn.functional.interpolate(prediction_u.view(1, d, H_d, W_d), size=(H, W), mode='nearest')
        if not probability_propagation:
            prediction_u = torch.argmax(prediction_u, 1).cpu()  # (1, H, W)
        if flip_pred:
            prediction_u = hflip(prediction_u)

        if probability_propagation:
            reduction = REDUCTIONS.get(reduction_str)
            prediction = reduction(prediction_o, prediction_u).cpu().half()
            prediction = torch.argmax(prediction, 1).cpu()  # (1, H, W)
        else:
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


def inference_multimodel(model, additional_model, inference_loader, total_len, annotation_dir, last_video, save,
                         sigma_1, sigma_2, frame_range, ref_num, temperature, probability_propagation, reduction_str,
                         disable):
    global pred_visualize, label_history_a, feats_history_a, weight_sparse, weight_dense, label_history_o, feats_history_o, d, palette
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
                feats_history_o = model(input)
                feats_history_a = additional_model(input)
            first_annotation = annotation_dir / current_video / '00000.png'
            label_history, d, palette, weight_dense, weight_sparse = prepare_first_frame(
                current_video,
                save,
                first_annotation,
                sigma_1,
                sigma_2,
                inference_strategy='multimodel',
                probability_propagation=probability_propagation)
            frame_idx += 1
            last_video = current_video
            label_history_o = label_history
            label_history_a = label_history
            continue
        (_, _, H, W) = input.shape

        input = input.to(Config.DEVICE)
        with torch.cuda.amp.autocast():
            features_o = model(input)
            features_a = additional_model(input)

        (_, feature_dim, H_d, W_d) = features_o.shape
        prediction_o = predict(feats_history_o,
                               features_o[0],
                               label_history_o,
                               weight_dense,
                               weight_sparse,
                               frame_idx,
                               frame_range,
                               ref_num,
                               temperature,
                               probability_propagation)
        # Store all frames' features
        if probability_propagation:
            new_label_o = prediction_o.unsqueeze(1)
        else:
            new_label_o = index_to_onehot(torch.argmax(prediction_o, 0), d).unsqueeze(1)
        label_history_o = torch.cat((label_history_o, new_label_o), 1)
        feats_history_o = torch.cat((feats_history_o, features_o), 0)

        prediction_o = torch.nn.functional.interpolate(prediction_o.view(1, d, H_d, W_d), size=(H, W), mode='nearest')
        if not probability_propagation:
            prediction_o = torch.argmax(prediction_o, 1).cpu()  # (1, H, W)

        (_, feature_dim, H_d, W_d) = features_a.shape
        prediction_a = predict(feats_history_a,
                               features_a[0],
                               label_history_a,
                               weight_dense,
                               weight_sparse,
                               frame_idx,
                               frame_range,
                               ref_num,
                               temperature,
                               probability_propagation)
        # Store all frames' features
        if probability_propagation:
            new_label_a = prediction_a.unsqueeze(1)
        else:
            new_label_a = index_to_onehot(torch.argmax(prediction_a, 0), d).unsqueeze(1)
        label_history_a = torch.cat((label_history_a, new_label_a), 1)
        feats_history_a = torch.cat((feats_history_a, features_a), 0)

        prediction_a = torch.nn.functional.interpolate(prediction_a.view(1, d, H_d, W_d), size=(H, W), mode='nearest')
        if not probability_propagation:
            prediction_a = torch.argmax(prediction_a, 1).cpu()  # (1, H, W)

        if probability_propagation:
            reduction = REDUCTIONS.get(reduction_str)
            prediction = reduction(prediction_o, prediction_a).cpu().half()
            prediction = torch.argmax(prediction, 1).cpu()  # (1, H, W)
        else:
            prediction = torch.maximum(prediction_o, prediction_a).cpu().half()

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
                      frame_range, ref_num, temperature, probability_propagation, scale, disable):
    global pred_visualize, palette, feats_history, label_history, weight_dense, weight_sparse, d, current_video
    scales = [0.9, 1.0, scale]
    predictions = {}
    palettes = []
    for scale in scales:
        frame_idx = 0
        for i, (input, (current_video,)) in tqdm(enumerate(inference_loader), total=total_len, disable=disable):
            (_, _, H, W) = input.shape
            H_d = int(np.ceil(H * scale))
            W_d = int(np.ceil(W * scale))
            input = torch.nn.functional.interpolate(input, size=(H_d, W_d), mode='nearest').to(Config.DEVICE)
            if i != 0 and current_video != last_video:
                # save prediction
                pred_visualize = pred_visualize.cpu().numpy()
                if last_video not in predictions:
                    predictions[last_video] = []
                predictions[last_video].append(pred_visualize)
                frame_idx = 0
            if frame_idx == 0:
                with torch.cuda.amp.autocast():
                    feats_history = model(input)
                first_annotation = annotation_dir / current_video / '00000.png'
                label_history, d, palette, weight_dense, weight_sparse = prepare_first_frame(
                    current_video,
                    save,
                    first_annotation,
                    sigma_1,
                    sigma_2,
                    inference_strategy='3-scale',
                    probability_propagation=probability_propagation,
                    scale=scale)
                frame_idx += 1
                last_video = current_video
                palettes.append(palette)
                continue

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
                                 temperature,
                                 probability_propagation)
            # Store all frames' features
            if probability_propagation:
                new_label = prediction.unsqueeze(1)
            else:
                new_label = index_to_onehot(torch.argmax(prediction, 0), d).unsqueeze(1)
            label_history = torch.cat((label_history, new_label), 1)
            feats_history = torch.cat((feats_history, features), 0)

            prediction = torch.nn.functional.interpolate(prediction.view(1, d, H_d, W_d), size=(480, 910),
                                                         mode='nearest')
            prediction = torch.argmax(prediction, 1).cpu().type(torch.int8)  # (1, H, W)

            last_video = current_video
            frame_idx += 1

            if frame_idx == 2:
                pred_visualize = prediction
            else:
                pred_visualize = torch.cat((pred_visualize, prediction), 0)

        pred_visualize = pred_visualize.cpu().numpy()
        if current_video not in predictions:
            predictions[current_video] = []
        predictions[current_video].append(pred_visualize)
        pred_visualize = None

    for (video_name, frames), palette in tqdm(zip(predictions.items(), palettes), desc='Saving',
                                              total=len(predictions)):
        prediction = np.maximum(np.maximum(frames[0], frames[1]), frames[2])
        save_predictions(prediction, palette, save, video_name)
