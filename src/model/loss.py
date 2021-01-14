# -*- encoding: utf-8 -*-
# ! python3


import torch
from loguru import logger
from torch import nn
from torch.nn import functional as F

from src.config import Config


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

        prediction = prediction.softmax(dim=1).topk(k=1, dim=1).indices.squeeze(axis=1)
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

        negatives = self.sample_negatives(tensor, tensor_labels, anchors_labels)

        return anchors, positives, negatives

    def sample_negatives(self, tensor, labels, labels_to_omit):
        tensor = tensor.reshape(tensor.shape[0], -1, 256)
        labels = labels.reshape(labels.shape[0], -1)
        labels_to_omit = labels_to_omit.reshape(labels_to_omit.shape[0], -1)
        negatives = torch.zeros(size=(labels_to_omit.shape[0], labels_to_omit.shape[1], 256), device=Config.DEVICE)

        for batch_idx, batch_labels in enumerate(labels_to_omit):
            label_space = labels[batch_idx]
            for label_idx, label in enumerate(labels_to_omit[batch_idx]):
                feasible_features = tensor[batch_idx, label_space != label]
                if feasible_features.numel() == 0:
                    idx = torch.randint(high=tensor[batch_idx].shape[0], size=(1,), device=Config.DEVICE)
                    negative = tensor[batch_idx][idx].squeeze()
                else:  # todo: vypocitat cosine distance anchor -> feasible_features a vzit nejblizsi
                    idx = torch.randint(high=feasible_features.shape[0], size=(1,), device=Config.DEVICE)
                    negative = feasible_features[idx].squeeze()
                negatives[batch_idx, label_idx] = negative
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


def single_embedding_loss(anchor_point, pool):
    """
    Compute the loss from a pool of embedding points to a single anchor embedding point;
    as per paper, this computes the first term in the summation in Eq (1);
    for $x^a \in A$, this computes $min_{x^P \in P} ||f(x^a)-f(x^p)||^2_2$
    :param anchor_point:
    :param pool:
    :return:
    """
    return torch.min(torch.sum(torch.pow(torch.sub(pool, anchor_point), 2), dim=1))


def distance_matrix(x, y):
    """
    Computes distance matrix between two sets of embedding points
    shamelessly simplified version of https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3
    :param x: N x d tensor
    :param y: M x d tensor (M need not be same as N)
    :return: N x M distance matrix
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, min=0.0)
    # differences = x.unsqueeze(1) - y.unsqueeze(0)
    # distances = torch.sum(differences * differences, -1)
    # return distances


def validation_loss(anchor_points, positive_pool, negative_pool):
    """
    Computes validation loss as fraction of triplets where (negative sample, anchor point) is closer than (positive sample, anchor point)
    :param anchor_points: Nxd tensor representing N anchor points
    :param positive_pool: Mxd tensor representing M positive pool points
    :param negative_pool: Lxd tensor representing L negative pool points
    :return: float validation loss
    """
    # TODO: Find replacement for repeat() since it blows up GPU memory even in Pytorch eval mode.
    positive_distances = distance_matrix(anchor_points, positive_pool)  # N x M
    negative_distances = distance_matrix(anchor_points, negative_pool)  # N x L
    N, M = positive_distances.size()
    N, L = negative_distances.size()
    p_ = positive_distances.repeat(1, L)  # N x (M*L)
    n_ = negative_distances.repeat(1, M)  # N x (M*L)
    # For each anchor point, for each combination of positive, negative pair, count fraction of pairs
    # where the positive point is farther than negative point
    return torch.sum(torch.gt(p_, n_)).float() / float(N * M * L)


class MinTripletLoss(torch.nn.Module):
    def __init__(self, alpha=1):
        super().__init__()
        self._alpha = alpha

    def forward(self, anchor_points, positive_pool, negative_pool):
        positive_distances = distance_matrix(anchor_points, positive_pool)
        negative_distances = distance_matrix(anchor_points, negative_pool)
        losses = F.relu(
            torch.min(positive_distances, 1)[0].sum() - torch.min(negative_distances, 1)[0].sum() + self._alpha
        )
        return losses.mean()
