# -*- encoding: utf-8 -*-
# ! python3


from __future__ import annotations
from __future__ import generator_stop

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
import torch
from PIL import Image
from skimage.transform import resize
from torch.nn import DataParallel
from tqdm import tqdm, trange
from scipy import sparse
from loguru import logger

from src.config import Config
from src.model.predict import get_spatial_weight
from src.model.vos_net import VOSNet
from src.utils.datasets import InferenceDataset
from src.utils.utils import index_to_onehot


def load_annotation(path):
    annotation = Image.open(path)
    (H, W) = np.asarray(annotation).shape
    palette = annotation.getpalette()
    label = np.asarray(annotation)
    d = np.max(label) + 1
    label = torch.tensor(label, dtype=torch.long)
    label_1hot = index_to_onehot(label.view(-1), d).reshape(1, d, H, W)
    size = np.round(np.array([H, W]) * Config.SCALE).astype(np.int)
    labels = torch.nn.functional.interpolate(label_1hot, size=tuple(size.data))
    return labels.to(torch.long).numpy(), palette


def generate_features(save_path, checkpoint_path, data_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model = VOSNet(model='resnet50')
    model = DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'])

    inference_dataset = InferenceDataset(data_path)
    inference_loader = torch.utils.data.DataLoader(inference_dataset, batch_size=1, shuffle=False, num_workers=4)
    features = []
    for img, _ in tqdm(inference_loader):
        features_tensor: torch.Tensor = model(img)
        features.append(features_tensor.detach().numpy())
    features = np.array(features)
    features = features.squeeze()
    np.savez(save_path, features=features)
    return features


def save_predictions(save_path: Path, predictions: np.array, palette, gif=True, show=False):
    save_path.mkdir(exist_ok=True, parents=True)
    if gif:
        def create_img(prediction):
            im = Image.fromarray(np.uint8(prediction), mode='P')
            im.putpalette(palette)
            return im

        image = create_img(predictions[0])
        image.save(save_path / f'video.gif', save_all=True,
                   append_images=list(map(lambda p: create_img(p), predictions[1:])), loop=0,
                   duration=250)
    else:
        for i, frame in enumerate(predictions):
            if show:
                sns.heatmap(frame)
                plt.show()
            image = Image.fromarray(np.uint8(frame), mode='P')
            image.putpalette(palette)
            image.save(save_path / f'{str(i).rjust(5, "0")}.png')


def get_features(features_save_path, checkpoint_path, data_dir):
    if not features_save_path.exists():
        features = generate_features(features_save_path, checkpoint_path, data_dir)
    else:
        features = np.load(features_save_path)['features']
    return features


def softmax(X, axis=None):
    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
    # finally: divide elementwise
    p = y / ax_sum
    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()
    return p


def get_similarity_matrix(similarity_save_path, features, spatial_weight, K=150):
    if similarity_save_path.exists():
        logger.info('Loading similarity matrix.')
        similarity = sparse.load_npz(similarity_save_path).tolil()
    else:
        (N, dims, w, h) = features.shape
        mat_size = N * w * h
        similarity = sparse.lil_matrix((mat_size, mat_size), dtype=np.float16)
        for i in trange(features.shape[0]):
            for j in range(features.shape[0]):
                a = np.transpose(features[i], axes=[1, 2, 0]).reshape((w * h, dims))
                b = features[j].reshape((dims, w * h))
                c = a @ b
                c = (softmax(c, axis=0) * spatial_weight).T
                shape = c.shape
                similarity[j * shape[1]:j * shape[1] + shape[1], i * shape[0]:i * shape[0] + shape[0]] = c.astype(
                    np.float16)

        sparse.save_npz(similarity_save_path, similarity.tocsr(), compressed=True)

    if K != -1:
        logger.info(f'Selecting top {K=} from each row.')
        for i in trange(similarity.shape[0]):
            row = similarity[i].toarray().squeeze()
            indices = np.argpartition(row, -K)[-K:]
            values = row[indices]
            row[:] = 0
            row[indices] = values
            similarity[i] = row

    logger.info('Returning sparse matrix.')
    return similarity.tocsr()


def eq_3(frames, similarity: sparse.csr_matrix, labels, alpha=0.99):
    frames -= 1
    all_frames = np.zeros((frames * labels.shape[0] * labels.shape[1], 1), dtype=np.float32)
    labels = labels.reshape(labels.shape[0] * labels.shape[1], -1).astype(np.float32)
    all_frames = np.vstack((labels, all_frames))

    y_old = all_frames

    iter = 10
    for _ in tqdm(range(iter), desc='Computing y_new.'):
        y_new = alpha * (similarity @ y_old) + (1 - alpha) * all_frames
        y_old = y_new

    return y_old


def main(K):
    features_save_path = Path('features.npz')
    similarity_save_path = Path('similarity.npz')
    checkpoint_path = Path('/home/hynek/projects/checkpoint.pth.tar')
    data_dir = Path('/home/hynek/skola/FEL/5. semestr/test/480p/')
    annotation_path = Path('/home/hynek/skola/FEL/5. semestr/test/annot/00000.png')
    predictions_save_path = Path(f'/home/hynek/VOS_saves/{K=}')
    predictions_save_path.mkdir(parents=True, exist_ok=True)
    labels, palette = load_annotation(annotation_path)
    show = False

    logger.info('Loading features.')
    features = get_features(features_save_path, checkpoint_path, data_dir)
    logger.info('Getting spatial weight.')
    spatial_weight = get_spatial_weight(features.shape[2:], sigma=8).numpy()
    logger.info('Getting similarity matrix.')
    similarity_matrix = get_similarity_matrix(similarity_save_path, features, spatial_weight, K=K)

    logger.info('Processing predictions.')
    frames = np.zeros(shape=(features.shape[0], labels.shape[1], 480, 854))
    for i, curr_labels in enumerate(labels[0]):
        predicted_frames = eq_3(features.shape[0], similarity_matrix,
                                curr_labels)
        predicted_frames = predicted_frames.reshape((features.shape[0], labels.shape[2], labels.shape[3]))
        for j, frame in enumerate(predicted_frames):
            frame = resize(frame, (480, 854))
            frames[j][i] = frame
            if show:
                plt.figure()
                sns.heatmap(frame)
    frames = np.argmax(frames, axis=1)

    logger.info('Saving predictions.')
    save_predictions(predictions_save_path, frames, palette, show=show)


if __name__ == '__main__':
    main(-1)
