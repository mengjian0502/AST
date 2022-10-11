"""
Prune largest, separately regrow
"""
import torch
import torch.nn as nn
from models import SparsConv2d, SparsLinear
from typing import List
from .sparse import Mask
from models import SparsConv2d, SparsLinear

class MultiRegMask(Mask):
    def __init__(self, model: nn.Module, optimizer=None, prune_rate=0.3, prune_rate_decay=None, args=None, train_loader=None, slist: List = None, Itertrain=False):
        super().__init__(model, optimizer, prune_rate, prune_rate_decay, args, train_loader, slist, Itertrain)

        # masks with highest sparsity
        self.background = {}

    def switch(self, n):
        self.final_density = self.slist[n]

        # restate the mask and update the background (if necessary)
        for name, module in self.model.named_modules():
            if isinstance(module, (SparsConv2d, SparsLinear)):
                self.masks[name] = module.mask
                if n == 0:
                    self.background[name] = module.mask

    def _background_stats(self):
        self.sub_name2nonzeros = {}
        self.sub_bname2zeros = {}

        self.name2nonzeros = {}
        self.name2zeros = {}

        for name, mask in self.background.items():
            self.name2nonzeros[name] = mask.sum().item()
            self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]

        for name, mask in self.masks.items():
            self.sub_name2nonzeros[name] = mask.sum().item()
            self.sub_bname2zeros[name] = mask.numel() - self.sub_name2nonzeros[name]
            

    def pruning(self, step, bidx, n):
        """
        Step 1: Scheduled pruning
        """
        curr_prune_iter = int(step / self.prune_every_k_steps)
        final_iter = int((self.args.final_prune_epoch * len(self.loader)*self.args.multiplier) / self.prune_every_k_steps)
        ini_iter = int((self.args.init_prune_epoch * len(self.loader)*self.args.multiplier) / self.prune_every_k_steps)
        total_prune_iter = final_iter - ini_iter

        # message
        print(f'Pruning Progress is {curr_prune_iter - ini_iter} / {total_prune_iter}')
        if curr_prune_iter >= ini_iter and curr_prune_iter <= final_iter:
            # update sparsity schedule
            ramping_decay = (1 - ((curr_prune_iter - ini_iter) / total_prune_iter)) ** 3
            self.curr_prune_rate = (1 - self.args.init_density) + (self.args.init_density - self.final_density) * (1 - ramping_decay)

            # magnitude score
            mp_scores = self.collect_score()
            num_params_to_keep = int(len(mp_scores) * (1 - self.curr_prune_rate))
            topkscore, _ = torch.topk(mp_scores, num_params_to_keep, sorted=True)
            threshold = topkscore[-1]

            # update and apply the masks
            self.update_mask(threshold)
            if n == 0:
                self.apply_mask()
        else:
            self.curr_prune_rate = 0.
        
        # sparsity
        total_params, spars_params = self._param_stats()
        sparsity = spars_params / total_params
        print("Sparsity after pruning at step [{}] = {:.3f}".format(bidx, sparsity*100))

    def prune_and_regrow(self, step):
        self._background_stats()

        for n, m in self.model.named_modules():
            if isinstance(m, (SparsConv2d, SparsLinear)):
                weight = m.weight
                
                # candidate selection based on the background mask (largest sparsity)
                new_mask = self.magnitude_death(weight, n)

                extra = self.sub_name2nonzeros[n] - self.name2nonzeros[n]
                self.pruning_count[n] = int(self.name2nonzeros[n] - new_mask.sum().item()) + int(extra)
                
                # if self.final_density == 0.1:
                #     import pdb;pdb.set_trace()
                
                new_mask = self.gradient_growth(new_mask, self.pruning_count[n], weight)

                # apply mask
                m.mask.data = new_mask.clone()
                
                # record mask
                self.masks[n] = new_mask
        
        # sparsity
        total_params, spars_params = self._param_stats()
        sparsity = spars_params / total_params
        print("Sparsity [{:.3f}] after regrow at step [{}] = {:3f}".format(self.final_density, step, sparsity*100))

    def step(self, n, bidx):
        self.prune_rate_decay.step()
        self.prune_rate = self.prune_rate_decay.get_dr()
        self.steps[n] += 1
        
        if self.iter:
            prune_step = self.steps[n] * len(self.slist) if bidx < len(self.slist) else (self.steps[n]-1) * len(self.slist)
        else:
            prune_step = self.steps[n]
    
        if prune_step >= (self.args.init_prune_epoch * len(self.loader)*self.args.multiplier) and prune_step % self.prune_every_k_steps == 0:
            self.pruning(prune_step, bidx, n)
            self.prune_and_regrow(bidx)