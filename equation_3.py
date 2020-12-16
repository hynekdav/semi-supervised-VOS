# -*- encoding: utf-8 -*-
# ! python3


from __future__ import annotations
from __future__ import generator_stop

import asyncio
import time
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from loguru import logger
from scipy import sparse
from skimage.transform import resize
from torch.nn import DataParallel
from torchvision import transforms
from tqdm import tqdm, trange

from src.config import Config
from src.model.predict import get_spatial_weight, get_temporal_weight, get_descriptor_weight
from src.model.vos_net import VOSNet
from src.utils.utils import index_to_onehot


def load_annotation(path):
    annotation = Image.open(path)
    (H, W) = np.asarray(annotation).shape
    palette = annotation.getpalette()
    label = np.asarray(annotation)
    d = np.max(label) + 1
    label = torch.tensor(label, dtype=torch.long, device=Config.DEVICE)
    label_1hot = index_to_onehot(label.view(-1), d).reshape(1, d, H, W)
    size = np.round(np.array([H, W]) * Config.SCALE).astype(np.int)
    labels = torch.nn.functional.interpolate(label_1hot.to(Config.DEVICE), size=tuple(size.data))
    return labels.to(torch.long).cpu().numpy(), palette


def generate_features(save_path, checkpoint_path, data_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model = VOSNet(model='resnet50')
    model = DataParallel(model)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    rgb_normalize = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])])
    features = []
    for img_path in tqdm(sorted(list(data_path.glob('*.jpg')))):
        img = Image.open(img_path).convert('RGB')
        img = rgb_normalize(np.asarray(img)).unsqueeze(0)
        features_tensor: torch.Tensor = model(img)
        features.append(features_tensor.detach().cpu().numpy())
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
                   duration=125)
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


async def _top_k(data, row, k):
    """
    Helper function to process a single row of top_k
    """
    data, row = zip(*sorted(zip(data, row), reverse=True)[:k])
    return data, row


async def top_k(m, k):
    """
    Keep only the top k elements of each row in a csr_matrix
    """
    ml = m.tolil()

    tasks = [asyncio.create_task(_top_k(data, row, k)) for data, row in zip(ml.data, ml.rows)]
    for t in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        await t
    results = [t.result() for t in tasks]
    ml.data, ml.rows = zip(*results)

    return ml.tocsr()


def get_similarity_matrix(similarity_save_path, features, spatial_weight, K=150):
    if similarity_save_path.exists():
        logger.info('Loading similarity matrix.')
        similarity = sparse.load_npz(similarity_save_path).tolil()
    else:
        (N, dims, w, h) = features.shape
        mat_size = N * w * h
        similarity = sparse.lil_matrix((mat_size, mat_size))
        for i in trange(features.shape[0]):
            for j in range(features.shape[0]):
                a = np.transpose(features[i], axes=[1, 2, 0]).reshape((w * h, dims))
                b = features[j].reshape((dims, w * h))
                c = a @ b
                softmaxed = softmax(c, axis=0)
                # temporal_weight = get_temporal_weight(a, b, sigma=8)
                # descriptor_weight = get_descriptor_weight(c, p=3<)
                # c = (softmaxed * temporal_weight * descriptor_weight * spatial_weight).T
                c = (softmaxed * spatial_weight).T
                shape = c.shape
                similarity[j * shape[1]:j * shape[1] + shape[1], i * shape[0]:i * shape[0] + shape[0]] = c.astype(
                    np.float16)

        sparse.save_npz(similarity_save_path, similarity.tocsr(), compressed=True)

    if K != -1:
        logger.info(f'Selecting top {K=} from each row.')
        similarity = asyncio.run(top_k(similarity, K))
        # for i in trange(similarity.shape[0]):
        #     row = similarity[i].toarray().squeeze()
        #     indices = np.argpartition(row, -K)[-K:]
        #     values = row[indices]
        #     row[:] = 0
        #     row[indices] = values
        #     similarity[i] = row

    logger.info('Returning sparse matrix.')
    return similarity.tocsr()


def eq_3(frames, similarity: sparse.csr_matrix, labels, alpha=0.99):
    frames -= 1
    all_frames = np.zeros((frames * labels.shape[0] * labels.shape[1], 1), dtype=np.float64)
    labels = labels.reshape(labels.shape[0] * labels.shape[1], -1).astype(np.float32)
    all_frames = np.vstack((labels, all_frames))

    y_old = all_frames

    iter = 30
    for _ in tqdm(range(iter), desc='Computing y_new.'):
        y_new = alpha * (similarity @ y_old) + (1 - alpha) * all_frames
        y_old = y_new

    return y_old


@click.command(name='Eq_3')
@click.option('-K', 'K_value', type=click.INT, required=True, help='How many values of each row to take.')
@click.option('-d', '--data', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True,
              help='Images folder.')
@click.option('-a', '--annotation', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True,
              help='Annotation folder.')
@click.option('-c', '--checkpoint', type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True,
              help='Checkpoint path.')
@click.option('-s', '--save', type=click.Path(file_okay=False, dir_okay=True), required=True, help='Save path.')
@click.option('--show/--no-show', default=False, help='Show/no-show results.')
@click.option('--gif/--no-gif', default=False, help='Save gif/pngs.')
def equation_3(K_value, data, annotation, checkpoint, save, show, gif):
    K_value = int(K_value)
    data = Path(data)
    data_dir = data
    annotation_path = Path(annotation) / '00000.png'
    checkpoint_path = Path(checkpoint)
    predictions_save_path = Path(save) / f'K={K_value}'
    predictions_save_path.mkdir(parents=True, exist_ok=True)

    features_save_path = Path('features.npz')
    similarity_save_path = Path('similarity.npz')

    labels, palette = load_annotation(annotation_path)

    logger.info('Loading features.')
    features = get_features(features_save_path, checkpoint_path, data_dir)
    logger.info('Getting spatial weight.')
    spatial_weight = get_spatial_weight(features.shape[2:], sigma=8).cpu().numpy()
    logger.info('Getting similarity matrix.')
    similarity_matrix = get_similarity_matrix(similarity_save_path, features, spatial_weight, K=K_value)

    logger.info('Processing predictions.')
    start = time.time()
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
    end = time.time()
    logger.info(f'Processing took {end - start}s.')

    logger.info('Saving predictions.')
    save_predictions(predictions_save_path, frames, palette, show=show, gif=gif)


if __name__ == '__main__':
    equation_3()
    # (["1", "5", "10", "15", "25", "50", "150", "250", "500", "1000", "1500", "2000", "-1"])
