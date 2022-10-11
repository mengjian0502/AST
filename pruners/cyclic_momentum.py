"""
Model regrow based on gradient momentum
"""
import math
import torch
import torch.nn as nn
from typing import List
from collections import OrderedDict
from .sparse import Mask
from models import SparsConv2d, SparsLinear

class CyclicMomentumMask(Mask):
    def __init__(self, model: nn.Module, optimizer=None, prune_rate=0.3, prune_rate_decay=None, args=None, train_loader=None, 
            slist:List=None, Itertrain=False, gmomentum:float=0.99, offset=60):
        super().__init__(model, optimizer, prune_rate, prune_rate_decay, args, train_loader, slist, Itertrain)

        self.offset = offset
        self.gmomentum = gmomentum
        # flag of MA regrow
        self.moving_grad = False

        self.init_buffer()

    def init_buffer(self):
        # generate a dummy dictionary as a buffer
        state_copy = self.model.state_dict()
        self.buffer = OrderedDict()
        
        # initialize the buffer as zero
        for n, v in state_copy.items():
            if 'mask0' in n:
                name = n.replace('.mask0', '')
                self.buffer[name] = torch.zeros_like(v)
        print("[Debug] Buffer reset! Moving grad = {}".format(self.moving_grad))

    def update_momentum(self, cur_step: int, max_steps: int):
        # cur_m = (
        #     self.final_tau - (self.final_tau - self.gmomentum) * (math.cos(math.pi * cur_step / max_steps) + 1) / 2
        # )
        # if cur_step >= max_steps:
        #     cur_m = self.final_tau
        return self.gmomentum

    def get_gradient_for_weights(self, weight, name):
        g = super().get_gradient_for_weights(weight, name)
        if self.moving_grad:
            self.curr_gm = self.update_momentum(self.curr_prune_iter, self.total_prune_iter)

            running_g = self.buffer[name]
            running_g = running_g.mul(self.curr_gm) + (1-self.curr_gm) * g
        else:
            running_g = g
        self.buffer[name] = running_g
        return running_g
    
    def momentum_step(self):
        """
        Update the momentum between the subnets gap
        """
        for n, m in self.model.named_modules():
            if isinstance(m, (SparsConv2d, SparsLinear)):
                weight = m.weight
                running_g = self.get_gradient_for_weights(weight, n) 

    def gradient_growth(self, new_mask, total_regrowth, weight, name):
        # the moving averaged gradient during the past iterations between two subnets
        grad = self.buffer[name]

        # only grow the weights within the current sparsity range
        grad = grad*(new_mask==0).float()
        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        return new_mask

    def step(self, n, bidx, model, regrow=True):
        if bidx > self.prune_every_k_steps:
            self.moving_grad = True

        # update the model
        self.model = model
        
        self.prune_rate_decay.step()
        self.prune_rate = self.prune_rate_decay.get_dr()
        self.steps[n] += 1

        # moving gradient for regrow
        self.momentum_step()

        # offset between adjacent sparsity updates
        offset = n * self.offset

        if self.iter:
            prune_step = self.steps[n] * len(self.slist) if bidx < len(self.slist) else ((self.steps[n]-offset)-1) * len(self.slist)
        else:
            prune_step = self.steps[n]
    
        if prune_step >= (self.args.init_prune_epoch * len(self.loader)*self.args.multiplier) and prune_step % self.prune_every_k_steps == 0:
            if prune_step != 0:
                self.pruning(prune_step, bidx)
                if regrow:
                    self.prune_and_regrow(bidx)
                    self.init_buffer()