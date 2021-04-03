# -*- encoding: utf-8 -*-
# ! python3


import os
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np

from PIL import Image

from src.utils.utils import index_to_onehot

from src.config import Config


def predict(ref,
            target,
            ref_label,
            weight_dense,
            weight_sparse,
            frame_idx,
            range,
            ref_num,
            temperature):
    """
    The Predict Function.
    :param ref: (N, feature_dim, H, W)
    :param target: (feature_dim, H, W)
    :param ref_label: (d, N, H*W)
    :param weight_dense: (H*W, H*W)
    :param weight_sparse: (H*W, H*W)
    :param frame_idx:
    :return: (d, H, W)
    """
    # sample frames from history features
    d = ref_label.shape[0]
    sample_idx = sample_frames(frame_idx, range, ref_num)
    ref_selected = ref.index_select(0, sample_idx)
    ref_label_selected = ref_label.index_select(1, sample_idx).view(d, -1)

    # get similarity matrix
    (num_ref, feature_dim, H, W) = ref_selected.shape
    ref_selected = ref_selected.permute(0, 2, 3, 1).reshape(-1, feature_dim)
    target = target.reshape(feature_dim, -1)
    global_similarity = ref_selected.mm(target)

    # temperature step
    global_similarity *= temperature

    # softmax
    global_similarity = global_similarity.softmax(dim=0)

    # spatial weight and motion model
    global_similarity = global_similarity.contiguous().view(num_ref, H * W, H * W)
    if frame_idx > 15:
        # interval frames
        global_similarity[:-Config.CONTINUOUS_FRAME] *= weight_sparse
        # continuous frames
        global_similarity[-Config.CONTINUOUS_FRAME:] *= weight_dense
    else:
        global_similarity = global_similarity.mul(weight_dense)
    global_similarity = global_similarity.view(-1, H * W)

    # get prediction
    prediction = ref_label_selected.float().mm(global_similarity.float())
    return prediction


def sample_frames(frame_idx,
                  take_range,
                  num_refs):
    if frame_idx <= num_refs:
        sample_idx = list(range(frame_idx))
    else:
        dense_num = Config.CONTINUOUS_FRAME - 1
        sparse_num = num_refs - dense_num
        target_idx = frame_idx
        ref_end = target_idx - dense_num - 1
        ref_start = max(ref_end - take_range, 0)
        sample_idx = np.linspace(ref_start, ref_end, sparse_num).astype(np.int).tolist()
        for j in range(dense_num):
            sample_idx.append(target_idx - dense_num + j)

    return torch.Tensor(sample_idx).long().to(Config.DEVICE)


def get_labels(label, d, H, W, H_d, W_d):
    label_1hot = index_to_onehot(label.view(-1), d).reshape(1, d, H, W)
    label_1hot = F.interpolate(label_1hot, size=(H_d, W_d), mode='nearest')
    label_1hot = label_1hot.reshape(d, -1).unsqueeze(1)
    return label_1hot.type(torch.int32)


def prepare_first_frame(curr_video,
                        save_prediction,
                        annotation,
                        sigma1=8,
                        sigma2=21,
                        inference_strategy='single'):
    first_annotation = Image.open(annotation)
    (H, W) = np.asarray(first_annotation).shape
    H_d = int(np.ceil(H * Config.SCALE))
    W_d = int(np.ceil(W * Config.SCALE))
    palette = first_annotation.getpalette()
    label = np.asarray(first_annotation)
    d = np.max(label) + 1
    label = torch.Tensor(label).long().to(Config.DEVICE)  # (1, H, W)
    label_1hot = get_labels(label, d, H, W, H_d, W_d)

    weight_dense = get_spatial_weight((H_d, W_d), sigma1)
    weight_sparse = get_spatial_weight((H_d, W_d), sigma2)

    if save_prediction is not None:
        if not os.path.exists(save_prediction):
            os.makedirs(save_prediction)
        save_path = os.path.join(save_prediction, curr_video)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        first_annotation.save(os.path.join(save_path, '00000.png'))

    if inference_strategy == 'single':
        return label_1hot, d, palette, weight_dense, weight_sparse
    elif inference_strategy == 'hor-flip':
        label_1hot_flipped = get_labels(torch.fliplr(label), d, H, W, H_d, W_d)
        return label_1hot, label_1hot_flipped, d, palette, weight_dense, weight_sparse
    elif inference_strategy == 'ver-flip':
        label_1hot_flipped = get_labels(torch.flipud(label), d, H, W, H_d, W_d)
        return label_1hot, label_1hot_flipped, d, palette, weight_dense, weight_sparse
    elif inference_strategy == '2-scale':
        H_d_2 = int(np.ceil(H * Config.SCALE / 2))
        W_d_2 = int(np.ceil(W * Config.SCALE / 2))
        weight_dense_2 = get_spatial_weight((H_d_2, W_d_2), sigma1)
        weight_sparse_2 = get_spatial_weight((H_d_2, W_d_2), sigma2)
        label_1hot_2 = get_labels(label, d, H, W, H_d_2, W_d_2)
        return (label_1hot, label_1hot_2), d, palette, (weight_dense, weight_dense_2), (weight_sparse, weight_sparse_2)
    elif inference_strategy == '3-scale':
        pass

    return label_1hot, d, palette, weight_dense, weight_sparse


def get_spatial_weight(shape, sigma, t_loc: Optional[float] = None):
    """
    Get soft spatial weights for similarity matrix.
    :param shape: (H, W)
    :param sigma:
    :return: (H*W, H*W)
    """
    (H, W) = shape

    index_matrix = torch.arange(H * W, dtype=torch.long).reshape(H * W, 1).to(Config.DEVICE)
    index_matrix = torch.cat((index_matrix.div(float(W)), index_matrix % W), -1)  # (H*W, 2)
    d = index_matrix - index_matrix.unsqueeze(1)  # (H*W, H*W, 2)
    if t_loc is not None:
        d[d < t_loc] = 0.0
    d = d.float().pow(2).sum(-1)  # (H*W, H*W)
    w = (- d / sigma ** 2).exp()

    return w


def get_descriptor_weight(array: np.array, p: float = 0.5):
    pow = np.power(array, p)
    return np.sign(pow) * np.abs(pow)


def get_temporal_weight(frame_1: np.array, frame_2: np.array, sigma, t_temp: Optional[float] = None):
    d = frame_1 - frame_2.T
    if t_temp is not None:
        d[d < t_temp] = 0.0
    d = np.sum(np.power(d, 2), axis=-1)
    w = np.exp(-d / sigma ** 2)

    return w
