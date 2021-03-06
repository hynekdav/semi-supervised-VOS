# -*- encoding: utf-8 -*-
# ! python3


from __future__ import annotations
from __future__ import generator_stop

from pathlib import Path

import numpy as np
import torch


class EarlyStopping(object):
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, directory, *, patience=7, verbose=False, delta=0, path='model.pth.tar',
                 trace_func=print):
        """
        Args:
            directory (str): Path for the checkpoint to be saved to.
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Template for the checkpoint file.
                            Default: 'early-stopping-{:03d}-{:.5f}.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.directory = Path(directory)
        self.trace_func = trace_func

    def __call__(self, val_loss, epoch, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, epoch, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                self.trace_func(f'Best epoch was {self.best_epoch} with {self.val_loss_min}.')
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, epoch, model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, val_loss, epoch, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), str(self.directory / self.path))
        self.val_loss_min = val_loss
        self.best_epoch = epoch
