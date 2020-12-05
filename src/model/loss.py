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


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class SupervisedNTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = labels.unsqueeze(-1)

        mask = torch.eq(labels, labels.T).float().to(device)
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.ones_like(mask) - torch.eye(batch_size)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(torch.sum(exp_logits, dim=1, keepdim=True))

        mean_log_prob_pos = torch.sum(mask * log_prob, dim=1) / torch.sum(mask, dim=1)

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = torch.mean(loss[~torch.isnan(loss)])

        return loss
