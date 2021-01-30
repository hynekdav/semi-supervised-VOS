# -*- encoding: utf-8 -*-
# ! python3


from __future__ import annotations
from __future__ import generator_stop
from abc import ABC, abstractmethod

import torch
from torch import nn
import torch.nn.functional as F


def get_miner(miner_name):
    miners = {'default': KernelMiner(3, 3),
              'kernel_7x7': KernelMiner(7, 7),
              'temporal': TemporalMiner(),
              'one_back_one_ahead': OneBackOneAheadMiner()}
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
        similarity[labels != anchors_labels.unsqueeze(2)] = -1
        similarity[:, :, anchor_idx] = -1
        indices = similarity.argmax(dim=-1).reshape(similarity.shape[0] * similarity.shape[1])
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
        positive_candidates[different_labels] = -1

        negative_indices = torch.argmax(negative_candidates, dim=-1)
        positive_indices = torch.argmax(positive_candidates, dim=-1)

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
