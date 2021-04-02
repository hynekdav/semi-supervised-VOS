# -*- encoding: utf-8 -*-
# ! python3


from __future__ import annotations
from __future__ import generator_stop

import torch
import torch.nn.functional as F

from src.config import Config
from src.model.predict import prepare_first_frame, predict
from src.utils.utils import save_predictions, index_to_onehot
from tqdm import tqdm


def inference_single():
    pass


def inference_hor_flip(model, inference_loader, total_len, annotation_dir, last_video, save, sigma_1, sigma_2,
                       frame_range, ref_num, temperature, disable):
    frame_idx, video_idx = 0, 0
    for input, (current_video,) in tqdm(inference_loader, total=total_len, disable=disable):
        if current_video != last_video:
            # save prediction
            pred_visualize = pred_visualize.cpu().numpy()
            save_predictions(pred_visualize, palette, save, last_video)

            frame_idx = 0
            video_idx += 1
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
                flipped_labels=True)
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


def inference_ver_flip():
    pass


def inference_2_scale():
    pass


def inference_3_scale():
    pass
