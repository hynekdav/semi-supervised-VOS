# -*- encoding: utf-8 -*-
# ! python3


import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from src.config import Config
from src.model.triplet_miners import BaseTripletMiner


def batch_get_similarity_matrix(ref, target):
    """
    Get pixel-level similarity matrix.
    :param ref: (batchSize, num_ref, feature_dim, H, W)
    :param target: (batchSize, feature_dim, H, W)
    :return: (batchSize, num_ref*H*W, H*W)
    """
    (batchSize, num_ref, feature_dim, H, W) = ref.shape
    ref = ref.permute(0, 1, 3, 4, 2).reshape(batchSize, -1, feature_dim)
    target = target.reshape(batchSize, feature_dim, -1)
    T = ref.bmm(target)
    return T


def batch_global_predict(global_similarity, ref_label):
    """
    Get global prediction.
    :param global_similarity: (batchSize, num_ref*H*W, H*W)
    :param ref_label: onehot form (batchSize, num_ref, d, H, W)
    :return: (batchSize, d, H, W)
    """
    (batchSize, num_ref, d, H, W) = ref_label.shape
    ref_label = ref_label.transpose(1, 2).reshape(batchSize, d, -1)
    return ref_label.bmm(global_similarity).reshape(batchSize, d, H, W)


class CrossEntropy(nn.Module):
    def __init__(self, temperature=1.0):
        super(CrossEntropy, self).__init__()
        self.temperature = temperature
        self.nllloss = nn.NLLLoss()

    def forward(self, ref, target, ref_label, target_label):
        """
        let Nt = num of target pixels, Nr = num of ref pixels
        :param ref: (batchSize, num_ref, feature_dim, H, W)
        :param target: (batchSize, feature_dim, H, W)
        :param ref_label: label for reference pixels
                         (batchSize, num_ref, d, H, W)
        :param target_label: label for target pixels (ground truth)
                            (batchSize, H, W)
        """
        global_similarity = batch_get_similarity_matrix(ref, target)
        global_similarity = global_similarity * self.temperature
        global_similarity = global_similarity.softmax(dim=1)

        prediction = batch_global_predict(global_similarity, ref_label)
        prediction = torch.log(prediction + 1e-14)
        loss = self.nllloss(prediction, target_label)

        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.nllloss = nn.NLLLoss()
        self.contrastive_loss = nn.CosineEmbeddingLoss()

    def forward(self, ref, target, ref_label, target_label):
        """
        let Nt = num of target pixels, Nr = num of ref pixels
        :param ref: (batchSize, num_ref, feature_dim, H, W)
        :param target: (batchSize, feature_dim, H, W)
        :param ref_label: label for reference pixels
                         (batchSize, num_ref, d, H, W)
        :param target_label: label for target pixels (ground truth)
                            (batchSize, H, W)
        """
        global_similarity = batch_get_similarity_matrix(ref, target)
        global_similarity = global_similarity * self.temperature
        global_similarity = global_similarity.softmax(dim=1)

        prediction = batch_global_predict(global_similarity, ref_label)
        prediction = torch.log(prediction + 1e-14)
        loss = self.nllloss(prediction, target_label)

        prediction = prediction.softmax(dim=1).argmax(dim=1)
        y = torch.ones(size=prediction.shape, device=Config.DEVICE)
        y[prediction != target_label] = -1
        metric_loss = self.contrastive_loss(ref[:, -1, :, :], target, y)

        return loss + metric_loss


class TripletLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(TripletLoss, self).__init__()
        self.temperature = temperature
        self.nllloss = nn.NLLLoss()
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=nn.CosineSimilarity())

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
        # labels_to_omit = labels_to_omit.reshape(labels_to_omit.shape[0], -1)
        # negatives = torch.zeros(size=(labels_to_omit.shape[0], labels_to_omit.shape[1], 256), device=Config.DEVICE)

        dist = 1 - torch.cdist(F.normalize(anchors, p=1, dim=-1), F.normalize(tensor, p=1, dim=-1), p=2)
        invalid = torch.cdist(labels_to_omit.unsqueeze(-1).float(), labels.unsqueeze(-1).float(), p=1).long() == 0
        dist[invalid] = -1
        max_indices = torch.argmax(dist, dim=-1)
        negatives = self.batched_index_select(tensor, 1, max_indices)

        # for batch_idx, batch_labels in enumerate(labels_to_omit):
        #     label_space = labels[batch_idx]
        #     for label_idx, label in enumerate(labels_to_omit[batch_idx]):
        #         feasible_features = tensor[batch_idx, label_space != label]
        #         if feasible_features.numel() == 0:
        #             idx = torch.randint(high=tensor[batch_idx].shape[0], size=(1,), device=Config.DEVICE)
        #             negative = tensor[batch_idx][idx].squeeze()
        #         else:  # todo: vypocitat cosine distance anchor -> feasible_features a vzit nejblizsi
        #             idx = torch.randint(high=feasible_features.shape[0], size=(1,), device=Config.DEVICE)
        #             negative = feasible_features[idx].squeeze()
        #         negatives[batch_idx, label_idx] = negative
        return negatives

    def forward(self, ref, target, ref_label, target_label):
        """
        let Nt = num of target pixels, Nr = num of ref pixels
        :param ref: (batchSize, num_ref, feature_dim, H, W)
        :param target: (batchSize, feature_dim, H, W)
        :param ref_label: label for reference pixels
                         (batchSize, num_ref, d, H, W)
        :param target_label: label for target pixels (ground truth)
                            (batchSize, H, W)
        """
        global_similarity = batch_get_similarity_matrix(ref, target)
        global_similarity = global_similarity * self.temperature
        global_similarity = global_similarity.softmax(dim=1)

        prediction = batch_global_predict(global_similarity, ref_label)
        prediction = torch.log(prediction + 1e-14)
        loss = self.nllloss(prediction, target_label)

        anchors, positives, negatives = self.get_triplets(target, target_label)

        metric_loss = self.triplet_loss(anchors, positives, negatives)

        return loss + metric_loss


class TripletLossWithMiner(nn.Module):
    def __init__(self, miner: BaseTripletMiner, *, margin=1.0, weights=(0.25, 0.75), temperature=1.0):
        super(TripletLossWithMiner, self).__init__()
        assert np.allclose(np.sum(weights), np.ones(shape=(1,)))
        assert len(weights) == 2
        self._cross_entropy = CrossEntropy(temperature=temperature)
        self._triplet_loss = nn.TripletMarginWithDistanceLoss(margin=margin, distance_function=nn.CosineSimilarity())
        self._miner = miner
        self._weights = weights

    def forward(self, ref, target, ref_label, target_label, extra_embeddings=None, extra_labels=None):
        """
        let Nt = num of target pixels, Nr = num of ref pixels
        :param ref: (batchSize, num_ref, feature_dim, H, W)
        :param target: (batchSize, feature_dim, H, W)
        :param ref_label: label for reference pixels
                         (batchSize, num_ref, d, H, W)
        :param target_label: label for target pixels (ground truth)
                            (batchSize, H, W)
        """
        cross_entropy_loss = self._cross_entropy(ref, target, ref_label, target_label)

        if extra_embeddings is not None and extra_labels is not None:
            target = extra_embeddings
            target_label = extra_labels
        anchors, positives, negatives = self._miner.get_triplets(target, target_label)

        metric_loss = self._triplet_loss(anchors, positives, negatives)

        return cross_entropy_loss * self._weights[0] + metric_loss * self._weights[1]


class FocalLoss(nn.Module):
    def __init__(self, gamma=0.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.nll_loss = nn.NLLLoss()

    def forward(self, ref, target, ref_label, target_label):
        """
        let Nt = num of target pixels, Nr = num of ref pixels
        :param ref: (batchSize, num_ref, feature_dim, H, W)
        :param target: (batchSize, feature_dim, H, W)
        :param ref_label: label for reference pixels
                         (batchSize, num_ref, d, H, W)
        :param target_label: label for target pixels (ground truth)
                            (batchSize, H, W)
        """
        global_similarity = batch_get_similarity_matrix(ref, target)
        global_similarity = global_similarity.softmax(dim=1)
        prediction = batch_global_predict(global_similarity, ref_label)

        if prediction.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = prediction.shape[1]
            prediction = prediction.permute(0, *range(2, prediction.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            target_label = target_label.reshape(-1)

        log_p = F.log_softmax(prediction, dim=-1)
        ce = self.nll_loss(log_p, target_label)

        # get true class column from each row
        all_rows = torch.arange(len(prediction))
        log_pt = log_p[all_rows, target_label]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
