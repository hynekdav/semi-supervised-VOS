# -*- encoding: utf-8 -*-
# ! python3


from __future__ import annotations
from __future__ import generator_stop

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import DataParallel
from tqdm import tqdm, trange

from src.config import Config
from src.model.predict import get_spatial_weight
from src.model.vos_net import VOSNet
from src.utils.datasets import InferenceDataset
from src.utils.utils import index_to_onehot

from scipy.linalg import fractional_matrix_power

from sklearn.preprocessing import binarize
import seaborn as sns
import matplotlib.pyplot as plt


def load_annotation(path):
    annotation = Image.open(path)
    (H, W) = np.asarray(annotation).shape
    palette = annotation.getpalette()
    label = np.asarray(annotation)
    d = np.max(label) + 1
    label = torch.tensor(label, dtype=torch.long)
    label_1hot = index_to_onehot(label.view(-1), d).reshape(1, d, H, W)
    # labels = label_1hot.long().numpy()
    labels = torch.nn.functional.interpolate(label_1hot, scale_factor=1 / 7.981308411)  # Hack for now Config.SCALE)
    return labels.to(torch.long), palette


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
        # features_tensor = torch.mean(features_tensor, dim=1,
        #                              keepdim=True)
        features.append(features_tensor.detach().numpy())
    features = np.array(features)
    features = features.squeeze()
    print(features.shape)
    np.savez(save_path, features=features)
    return features


def save_predictions(save_path: Path, predictions: np.array, palette, show=True):
    save_path.mkdir(exist_ok=True, parents=True)
    for i, frame in enumerate(predictions):
        if show:
            sns.heatmap(frame)
            plt.show()
        image = Image.fromarray(np.uint8(frame), mode='P')
        image.putpalette(palette)
        image.save(save_path / f'{str(i).rjust(5, "0")}.png')


def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def get_features(features_save_path, checkpoint_path, data_dir):
    if not features_save_path.exists():
        features = generate_features(features_save_path, checkpoint_path, data_dir)
    else:
        features = np.load(features_save_path)['features']
    return torch.tensor(features)


def get_similarity_matrix(similarity_save_path, features, spatial_weight, K=150):
    if similarity_save_path.exists():
        similarity = np.load(similarity_save_path)['similarity']
    else:
        (N, dims, w, h) = features.shape
        similarity = np.array(())
        for i in trange(features.shape[0]):
            row = []
            for j in range(features.shape[0]):
                a = features[i].permute(1, 2, 0).reshape(-1, dims)
                b = features[j].reshape(dims, -1)
                c = a.mm(b)
                c = c.softmax(dim=0) * spatial_weight
                c = c.view(-1, h * w)
                row.append(c.numpy())  # torch.cat((row, c.type(torch.float16)), 0)
            similarity = np.vstack((similarity, np.hstack(row))) if similarity.size != 0 else np.hstack(row)
        np.savez(similarity_save_path, similarity=similarity)
    similarity = torch.tensor(similarity)

    if K != -1:
        for i in range(similarity.shape[0]):
            topk = torch.topk(similarity[i], k=K, sorted=False)
            similarity[i] = torch.zeros(size=(1, similarity.shape[1]))
            similarity[i] = similarity[i].scatter(0, topk.indices, topk.values)

    return similarity


def eq_3(frames, similarity: torch.Tensor, labels, alpha=0.99):
    frames -= 1
    all_frames = torch.zeros(size=(frames * labels.shape[0] * labels.shape[1], 1), dtype=torch.float)
    labels = labels.reshape(labels.shape[0] * labels.shape[1], -1).to(torch.float)
    all_frames = torch.cat((labels, all_frames), dim=0)

    similarity = similarity.t()

    y_old = all_frames

    iter = 30
    for _ in tqdm(range(iter), desc='Computing y_new.'):
        y_new = alpha * (similarity.mm(y_old)) + (1 - alpha) * all_frames
        y_old = y_new

    return y_old


def main():
    # udelat matrix W, ktera bude udavat similaritu mezi jednotlivyma framama

    features_save_path = Path('features.npz')
    similarity_save_path = Path('similarity.npz')
    checkpoint_path = Path('/home/hynek/projects/checkpoint.pth.tar')
    data_dir = Path('/home/hynek/skola/FEL/5. semestr/test/480p/')
    annotation_path = Path('/home/hynek/skola/FEL/5. semestr/test/annot/00000.png')
    predictions_save_path = Path('/home/hynek/VOS_saves/')
    labels, palette = load_annotation(annotation_path)

    features = get_features(features_save_path, checkpoint_path, data_dir)
    spatial_weight = get_spatial_weight(features.shape[2:], sigma=8)
    similarity_matrix = get_similarity_matrix(similarity_save_path, features, spatial_weight, K=10)
    C = features.shape[0] * features.shape[2] * features.shape[3]

    frames = np.zeros((features.shape[0], labels.shape[1], labels.shape[2]))
    for i, curr_labels in enumerate(labels[0]):
        predicted_frames = eq_3(features.shape[0], similarity_matrix,
                                curr_labels)  # equation_3(features, np.array(curr_labels)) * (i + 1)
        predicted_frames = predicted_frames.reshape(4, 60, 107)
        for frame in predicted_frames:
            sns.heatmap(frame)
            plt.show()
            break
        # frames += predicted_frames
    frames -= 1

    save_predictions(predictions_save_path, frames, palette)


if __name__ == '__main__':
    main()
