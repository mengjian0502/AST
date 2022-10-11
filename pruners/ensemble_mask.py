"""
Ensembled Masking
"""

import torch.nn as nn
from typing import List
from .sparse import Mask

class EnsembleMask(Mask):
    def __init__(self, model: nn.Module, optimizer=None, prune_rate=0.3, prune_rate_decay=None, args=None, train_loader=None, slist: List = None):
        super().__init__(model, optimizer, prune_rate, prune_rate_decay, args, train_loader, slist, False)

    def step(self, n, model, bidx):
        self.model = model
        
        # steps
        if n == 0:
            self.prune_rate_decay.step()
            self.prune_rate = self.prune_rate_decay.get_dr()
        self.steps[n] += 1

        if self.steps[n] >= (self.args.init_prune_epoch * len(self.loader)*self.args.multiplier) and self.steps[n] % self.prune_every_k_steps == 0:
            self.pruning(self.steps[n], bidx)
            self.prune_and_regrow(self.steps[n])
