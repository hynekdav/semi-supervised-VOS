# -*- encoding: utf-8 -*-
# ! python3


from __future__ import annotations
from __future__ import generator_stop
from abc import ABC, abstractmethod

import torch
from torch import nn
import torch.nn.functional as F


class BaseTripletMiner(ABC):

    @abstractmethod
    def get_triplets(self, embeddings, labels):
        pass


class DefaultTripletMiner(BaseTripletMiner):
    def __init__(self):
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def sample_patches(self, tensor):
        size, stride = 3, 3
        patches = tensor.unfold(2, size, stride).unfold(3, size, stride)
        patches = patches.reshape(tensor.shape[0], tensor.shape[1], -1, size * size)
        patches = patches.permute((0, 2, 3, 1))
        return patches

    def sample_patches_labels(self, tensor):
        size, stride = 3, 3
        patches = tensor.unfold(1, size, stride).unfold(2, size, stride)
        patches = patches.reshape(tensor.shape[0], -1, size * size)
        return patches

    def get_triplets(self, tensor, tensor_labels):
        patches = self.sample_patches(tensor)
        labels = self.sample_patches_labels(tensor_labels)
        anchors = patches[:, :, 4]  # anchor will always be 4-th element of 3x3 patch
        anchors_labels = labels[:, :, 4]

        similarity = self.cosine_similarity(anchors.unsqueeze(2), patches)
        similarity[labels != anchors_labels.unsqueeze(2)] = -1
        similarity[:, :, 4] = -1
        indices = similarity.argmax(dim=-1).reshape(similarity.shape[0] * similarity.shape[1])
        patches = patches.reshape(similarity.shape[0] * similarity.shape[1], 9, 256)
        positives = patches[torch.arange(patches.shape[0]), indices]
        positives = positives.reshape(similarity.shape[0], similarity.shape[1], -1)

        negatives = self.sample_negatives(anchors, tensor, tensor_labels, anchors_labels)

        return anchors, positives, negatives

    def batched_index_select(self, t, dim, inds):
        dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
        out = t.gather(dim, dummy)  # b x e x f
        return out

    def sample_negatives(self, anchors, tensor, labels, labels_to_omit):
        tensor = tensor.reshape(tensor.shape[0], -1, 256)
        labels = labels.reshape(labels.shape[0], -1)

        dist = 1 - torch.cdist(F.normalize(anchors, p=1, dim=-1), F.normalize(tensor, p=1, dim=-1), p=2)
        invalid = torch.cdist(labels_to_omit.unsqueeze(-1).float(), labels.unsqueeze(-1).float(), p=1).long() == 0
        dist[invalid] = -1
        max_indices = torch.argmax(dist, dim=-1)
        negatives = self.batched_index_select(tensor, 1, max_indices)

        return negatives
