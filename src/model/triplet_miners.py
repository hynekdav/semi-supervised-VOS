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
from skimage.morphology import skeletonize
import numpy as np

from src.config import Config


def get_miner(miner_name):
    miners = {'default': KernelMiner(3, 3),
              'kernel_7x7': KernelMiner(7, 7),
              'temporal': TemporalMiner(),
              'one_back_one_ahead': OneBackOneAheadMiner(),
              'euclidean': DistanceTransformationMiner(metric='euclidean'),
              'manhattan': DistanceTransformationMiner(metric='manhattan'),
              'chebyshev': DistanceTransformationMiner(metric='chessboard'),
              'skeleton': SkeletonMiner(),
              'skeleton_distance_transform': SkeletonWithDistanceTransformMiner(),
              'skeleton_temporal': SkeletonTemporalMiner(),
              'wrong_predictions': WrongPredictionsMiner()}
    return miners.get(miner_name)


def batched_index_select(t, dim, inds):
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy)  # b x e x f
    return out


class AbstractTripletMiner(ABC):
    def __init__(self):
        self._cosine_similarity = nn.CosineSimilarity(dim=-1)
        self._max_triplets = 0

    @abstractmethod
    def get_triplets(self, embeddings, labels, prediction):
        pass

    def limit_triplets(self, triplets):
        assert len(triplets) == 3
        assert len(triplets[0].shape) == 3
        if self._max_triplets == 0 or triplets[0].shape[1] <= self._max_triplets:
            return triplets
        batch_num = triplets[0].shape[0]

        indices = []
        for batch in range(batch_num):
            current_indices = torch.randperm(triplets[0].shape[1], device=Config.DEVICE)[:self._max_triplets]
            indices.append(current_indices)
        indices = torch.stack(indices)

        anchors = batched_index_select(triplets[0], dim=1, inds=indices)
        positives = batched_index_select(triplets[1], dim=1, inds=indices)
        negatives = batched_index_select(triplets[2], dim=1, inds=indices)

        return anchors, positives, negatives

    @property
    def max_triplets(self):
        return self._max_triplets

    @max_triplets.setter
    def max_triplets(self, new_value):
        self._max_triplets = new_value


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

    def get_triplets(self, tensor, tensor_labels, prediction):
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

        anchors, positives, negatives = self.limit_triplets((anchors, positives, negatives))
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
    def get_triplets(self, embeddings, labels, prediction):
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

        anchors, positives, negatives = self.limit_triplets((anchors, positives, negatives))
        return anchors, positives, negatives


class OneBackOneAheadMiner(AbstractTripletMiner):
    def __init__(self):
        super().__init__()
        self.miner = TemporalMiner()

    def get_triplets(self, embeddings, labels, prediction):
        return self.miner.get_triplets(embeddings, labels, prediction)


class DistanceTransformationMiner(AbstractTripletMiner):
    def __init__(self, metric='euclidean'):
        super().__init__()
        available_metrics = {'euclidean', 'manhattan', 'taxicab', 'cityblock', 'chessboard'}
        assert metric in available_metrics
        self._distance_transformation = lambda n: n
        if metric == 'euclidean':
            self._distance_transformation = ndimage.distance_transform_edt
        else:
            self._distance_transformation = functools.partial(ndimage.distance_transform_cdt, metric=metric)

    def get_triplets(self, batched_embeddings, batched_labels, prediction):
        all_anchors, all_positives, all_negatives = [], [], []

        for embeddings, labels in zip(batched_embeddings, batched_labels):

            unique_labels = np.unique(labels.cpu().numpy())

            anchors, positives, negatives = [], [], []
            for label in unique_labels:
                binary_mask = (labels == label).cpu().numpy().astype(np.int32)
                distances, indices = self._distance_transformation(binary_mask, return_indices=True)
                pixels_to_process = list(zip(*np.nonzero(distances)))

                distances = torch.from_numpy(distances).float()
                positive_candidates = embeddings[:, torch.logical_not(torch.isclose(distances, torch.tensor(0.0)))]
                positive_candidates = positive_candidates.permute((1, 0))

                normalized_embeddings = F.normalize(positive_candidates, dim=-1, p=2)
                similarities = 1 - torch.cdist(normalized_embeddings, normalized_embeddings, p=2)

                for idx, (i, j) in enumerate(pixels_to_process):
                    anchor = embeddings[:, i, j]
                    anchors.append(anchor)
                    x, y = indices[:, i, j]
                    negatives.append(embeddings[:, x, y])

                    if positive_candidates.numel() == 0:
                        positive_idx = np.random.randint(low=0, high=len(pixels_to_process))
                        x, y = pixels_to_process[positive_idx]
                        positives.append(embeddings[:, x, y])
                    else:
                        positive_idx = torch.argmin(similarities[idx], dim=0)
                        positives.append(positive_candidates[positive_idx])
            all_anchors.append(torch.stack(anchors))
            all_positives.append(torch.stack(positives))
            all_negatives.append(torch.stack(negatives))

        anchors = torch.stack(all_anchors)
        positives = torch.stack(all_positives)
        negatives = torch.stack(all_negatives)

        anchors, positives, negatives = self.limit_triplets((anchors, positives, negatives))
        return anchors, positives, negatives


class SkeletonMiner(AbstractTripletMiner):
    def get_triplets(self, batched_embeddings, batched_labels, prediction):
        all_anchors, all_positives, all_negatives = [], [], []

        for embeddings, labels in zip(batched_embeddings, batched_labels):

            unique_labels = np.unique(labels.cpu().numpy())
            anchors, positives, negatives = [], [], []
            for label in unique_labels:
                binary_mask = (labels == label).cpu().numpy().astype(np.int32)
                skeleton = skeletonize(binary_mask).astype(np.uint8)

                current_anchors = embeddings[:, skeleton == 1]
                current_anchors = current_anchors.permute((1, 0))

                positive_candidates = embeddings[:, binary_mask == 1]
                positive_candidates = positive_candidates.permute((1, 0))

                negative_candidates = embeddings[:, binary_mask == 0]
                negative_candidates = negative_candidates.permute((1, 0))

                if positive_candidates.numel() == 0 \
                        or negative_candidates.numel() == 0 \
                        or current_anchors.numel() == 0:
                    continue

                normalized_anchors = F.normalize(current_anchors, dim=-1, p=2)
                normalized_positives = F.normalize(positive_candidates, dim=-1, p=2)
                normalized_negatives = F.normalize(negative_candidates, dim=-1, p=2)

                positive_similarities = 1 - torch.cdist(normalized_anchors, normalized_positives, p=2)
                negative_similarities = 1 - torch.cdist(normalized_anchors, normalized_negatives, p=2)

                current_positives = positive_candidates[torch.argmin(positive_similarities, dim=-1)]
                current_negatives = negative_candidates[torch.argmax(negative_similarities, dim=-1)]

                anchors.append(current_anchors)
                positives.append(current_positives)
                negatives.append(current_negatives)

            if len(anchors) != 0:
                all_anchors.append(torch.cat(anchors))
                all_positives.append(torch.cat(positives))
                all_negatives.append(torch.cat(negatives))

        if len(all_anchors) == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])

        anchors = torch.stack(all_anchors)
        positives = torch.stack(all_positives)
        negatives = torch.stack(all_negatives)

        anchors, positives, negatives = self.limit_triplets((anchors, positives, negatives))
        return anchors, positives, negatives


class SkeletonWithDistanceTransformMiner(AbstractTripletMiner):
    def __init__(self, metric='manhattan'):
        super().__init__()
        available_metrics = {'euclidean', 'manhattan', 'taxicab', 'cityblock', 'chessboard'}
        assert metric in available_metrics
        self._distance_transformation = lambda n: n
        if metric == 'euclidean':
            self._distance_transformation = ndimage.distance_transform_edt
        else:
            self._distance_transformation = functools.partial(ndimage.distance_transform_cdt, metric=metric)

    def get_triplets(self, batched_embeddings, batched_labels, prediction):
        all_anchors, all_positives, all_negatives = [], [], []

        for embeddings, labels in zip(batched_embeddings, batched_labels):

            unique_labels = np.unique(labels.cpu().numpy())
            anchors, positives, negatives = [], [], []
            for label in unique_labels:
                binary_mask = (labels == label).cpu().numpy().astype(np.int32)

                skeleton = skeletonize(binary_mask).astype(np.uint8)
                distances, indices = self._distance_transformation(binary_mask, return_indices=True)
                anchor_indices = np.where(np.logical_and(distances != 0, skeleton == 1))

                current_anchors = embeddings[:, skeleton == 1]
                current_anchors = current_anchors.permute((1, 0))

                positive_candidates = embeddings[:, np.logical_and(binary_mask == 1, skeleton == 0)]
                positive_candidates = positive_candidates.permute((1, 0))

                if positive_candidates.numel() == 0 \
                        or current_anchors.numel() == 0:
                    continue

                current_negatives = []
                for i, j in zip(*anchor_indices):
                    x, y = indices[:, i, j]
                    negative = embeddings[:, x, y]
                    current_negatives.append(negative)
                current_negatives = torch.stack(current_negatives)

                normalized_anchors = F.normalize(current_anchors, dim=-1, p=2)
                normalized_positives = F.normalize(positive_candidates, dim=-1, p=2)

                positive_similarities = 1 - torch.cdist(normalized_anchors, normalized_positives, p=2)

                current_positives = positive_candidates[torch.argmin(positive_similarities, dim=-1)]

                anchors.append(current_anchors)
                positives.append(current_positives)
                negatives.append(current_negatives)

            if len(anchors) != 0:
                all_anchors.append(torch.cat(anchors))
                all_positives.append(torch.cat(positives))
                all_negatives.append(torch.cat(negatives))

        if len(all_anchors) == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])

        anchors = torch.stack(all_anchors)
        positives = torch.stack(all_positives)
        negatives = torch.stack(all_negatives)

        anchors, positives, negatives = self.limit_triplets((anchors, positives, negatives))
        return anchors, positives, negatives


class SkeletonTemporalMiner(AbstractTripletMiner):
    def __init__(self):
        super().__init__()
        self._miner = SkeletonMiner()

    def get_triplets(self, embeddings, labels, prediction):
        return self._miner.get_triplets(embeddings, labels, prediction)


class WrongPredictionsMiner(AbstractTripletMiner):
    def get_triplets(self, batched_embeddings, batched_labels, prediction):
        batched_prediction = prediction
        all_anchors, all_positives, all_negatives = [], [], []
        feature_dim = batched_embeddings.shape[1]

        for i, (embeddings, labels, prediction) in enumerate(
                zip(batched_embeddings, batched_labels, batched_prediction)):
            difference = (prediction != labels).reshape(-1)
            labels = labels.reshape(-1)
            embeddings = embeddings.permute((1, 2, 0)).reshape(-1, feature_dim)
            anchors = embeddings[difference]
            normalized_anchors = F.normalize(anchors, p=2)
            anchors_labels = labels[difference]
            unique_labels = torch.unique(anchors_labels)
            positives = torch.zeros(size=anchors.shape, dtype=anchors.dtype)
            negatives = torch.zeros(size=anchors.shape, dtype=anchors.dtype)

            for label in unique_labels:
                current_anchors = anchors_labels == label
                indexer = torch.nonzero(current_anchors).squeeze()
                current_anchors = normalized_anchors[current_anchors]
                positive_candidates = F.normalize(embeddings[labels == label], p=2)
                negative_candidates = F.normalize(embeddings[labels != label], p=2)

                positive_similarities = 1 - torch.cdist(current_anchors, positive_candidates, p=2)
                negative_similarities = 1 - torch.cdist(current_anchors, negative_candidates, p=2)

                current_positives = embeddings[torch.argmin(positive_similarities, dim=-1)]
                current_negatives = embeddings[torch.argmax(negative_similarities, dim=-1)]

                positives[indexer] = current_positives
                negatives[indexer] = current_negatives
            all_anchors.append(anchors)
            all_positives.append(positives)
            all_negatives.append(negatives)

        anchors = torch.cat(all_anchors)
        positives = torch.cat(all_positives)
        negatives = torch.cat(all_negatives)

        return anchors, positives, negatives
