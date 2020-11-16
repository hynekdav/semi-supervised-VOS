# -*- encoding: utf-8 -*-
# ! python3


import torch
from torch import nn
from torch.nn import functional as F


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
