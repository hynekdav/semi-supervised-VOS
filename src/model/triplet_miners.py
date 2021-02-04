# -*- encoding: utf-8 -*-
# ! python3


from __future__ import annotations
from __future__ import generator_stop

import functools
from abc import ABC, abstractmethod

import torch
from torch import nn
import torch.nn.functional as F

from scipy import ndimage
import numpy as np


def get_miner(miner_name):
    miners = {'default': KernelMiner(3, 3),
              'kernel_7x7': KernelMiner(7, 7),
              'temporal': TemporalMiner(),
              'one_back_one_ahead': OneBackOneAheadMiner(),
              'euclidean': DistanceTransformationMiner(metric='euclidean'),
              'manhattan': DistanceTransformationMiner(metric='manhattan'),
              'chebyshev': DistanceTransformationMiner(metric='chessboard')}
    return miners.get(miner_name)


def batched_index_select(t, dim, inds):
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy)  # b x e x f
    return out


class AbstractTripletMiner(ABC):
    def __init__(self):
        self._cosine_similarity = nn.CosineSimilarity(dim=-1)

    @abstractmethod
    def get_triplets(self, embeddings, labels):
        pass


class KernelMiner(AbstractTripletMiner):
    def __init__(self, kernel_size=3, stride=3):
        super().__init__()
        self._kernel_size = kernel_size
        self._stride = stride

    def sample_patches(self, tensor):
        patches = tensor.unfold(2, self._kernel_size, self._stride).unfold(3, self._kernel_size, self._stride)
        patches = patches.reshape(tensor.shape[0], tensor.shape[1], -1, self._kernel_size * self._kernel_size)
        patches = patches.permute((0, 2, 3, 1))
        return patches

    def sample_patches_labels(self, tensor):
        patches = tensor.unfold(1, self._kernel_size, self._stride).unfold(2, self._kernel_size, self._stride)
        patches = patches.reshape(tensor.shape[0], -1, self._kernel_size * self._kernel_size)
        return patches

    def get_triplets(self, tensor, tensor_labels):
        anchor_idx = (self._kernel_size * self._kernel_size) // 2
        patches = self.sample_patches(tensor)
        labels = self.sample_patches_labels(tensor_labels)
        anchors = patches[:, :, anchor_idx]
        anchors_labels = labels[:, :, anchor_idx]

        similarity = self._cosine_similarity(anchors.unsqueeze(2), patches)
        similarity[labels != anchors_labels.unsqueeze(2)] = 10
        similarity[:, :, anchor_idx] = 10
        indices = similarity.argmin(dim=-1).reshape(similarity.shape[0] * similarity.shape[1])
        patches = patches.reshape(similarity.shape[0] * similarity.shape[1], self._kernel_size * self._kernel_size, 256)
        positives = patches[torch.arange(patches.shape[0]), indices]
        positives = positives.reshape(similarity.shape[0], similarity.shape[1], -1)

        negatives = self.sample_negatives(anchors, tensor, tensor_labels, anchors_labels)

        return anchors, positives, negatives

    def sample_negatives(self, anchors, tensor, labels, labels_to_omit):
        tensor = tensor.reshape(tensor.shape[0], -1, 256)
        labels = labels.reshape(labels.shape[0], -1)

        dist = 1 - torch.cdist(F.normalize(anchors, p=2, dim=-1), F.normalize(tensor, p=2, dim=-1), p=2)
        invalid = torch.cdist(labels_to_omit.unsqueeze(-1).float(), labels.unsqueeze(-1).float(), p=1).long() == 0
        dist[invalid] = -1
        max_indices = torch.argmax(dist, dim=-1)
        negatives = batched_index_select(tensor, 1, max_indices)

        return negatives


class TemporalMiner(AbstractTripletMiner):
    def get_triplets(self, embeddings, labels):
        embeddings = embeddings.permute(0, 1, 3, 4, 2)
        (batch_size, _, _, _, features_size) = embeddings.shape
        last_frame_embeddings = embeddings[:, -1, ...].reshape(batch_size, -1, features_size)
        last_frame_labels = labels[:, -1, ...].reshape(batch_size, -1)

        candidate_embeddings = embeddings[:, 0:-1, ...].reshape(batch_size, -1, features_size)
        candidate_labels = labels[:, 0:-1, ...].reshape(batch_size, -1)

        similarity = 1 - torch.cdist(F.normalize(last_frame_embeddings, p=2, dim=-1),
                                     F.normalize(candidate_embeddings, p=2, dim=-1), p=2)
        indices_distances = torch.cdist(last_frame_labels.unsqueeze(-1).float(),
                                        candidate_labels.unsqueeze(-1).float(), p=1).long()
        same_labels = indices_distances == 0
        different_labels = indices_distances != 0

        negative_candidates = torch.clone(similarity)
        negative_candidates[same_labels] = -1

        positive_candidates = torch.clone(similarity)
        positive_candidates[different_labels] = 10

        negative_indices = torch.argmax(negative_candidates, dim=-1)
        positive_indices = torch.argmin(positive_candidates, dim=-1)

        negatives = batched_index_select(candidate_embeddings, 1, negative_indices)
        positives = batched_index_select(candidate_embeddings, 1, positive_indices)
        anchors = torch.clone(last_frame_embeddings)

        return anchors, positives, negatives


class OneBackOneAheadMiner(AbstractTripletMiner):
    def __init__(self):
        super().__init__()
        self.miner = TemporalMiner()

    def get_triplets(self, embeddings, labels):
        return self.miner.get_triplets(embeddings, labels)


class DistanceTransformationMiner(AbstractTripletMiner):
    def __init__(self, metric='euclidean', margin=0.1):
        super().__init__()
        available_metrics = {'euclidean', 'manhattan', 'taxicab', 'cityblock', 'chessboard'}
        assert metric in available_metrics
        self._distance_transformation = lambda n: n
        if metric == 'euclidean':
            self._distance_transformation = ndimage.distance_transform_edt
        else:
            self._distance_transformation = functools.partial(ndimage.distance_transform_cdt, metric=metric)
        self._margin = margin

    def get_triplets(self, batched_embeddings, batched_labels):
        all_anchors, all_positives, all_negatives = [], [], []

        for embeddings, labels in zip(batched_embeddings, batched_labels):

            unique_labels = np.unique(labels.cpu().numpy())

            anchors, positives, negatives = [], [], []
            for label in unique_labels:
                binary_mask = (labels == label).cpu().numpy().astype(np.int32)
                distances, indices = self._distance_transformation(binary_mask, return_indices=True)
                pixels_to_process = list(zip(*np.nonzero(distances)))
                for i, j in pixels_to_process:
                    anchors.append(embeddings[:, i, j])
                    x, y = indices[:, i, j]
                    negatives.append(embeddings[:, x, y])
                    # todo: better way to pick positives
                    idx = np.random.randint(low=0, high=len(pixels_to_process))
                    x, y = pixels_to_process[idx]
                    positives.append(embeddings[:, x, y])
                pass
            all_anchors.append(torch.stack(anchors))
            all_positives.append(torch.stack(positives))
            all_negatives.append(torch.stack(negatives))

        return torch.stack(all_anchors), torch.stack(all_positives), torch.stack(all_negatives)
