"""
Model regrow based on gradient momentum
"""
import math
import torch
import torch.nn as nn
from typing import List
from collections import OrderedDict
from .sparse import Mask

class MomentumMask(Mask):
    def __init__(self, model: nn.Module, optimizer=None, prune_rate=0.3, prune_rate_decay=None, args=None, train_loader=None, slist: List = None, Itertrain=False, gmomentum:float=0.99):
        super().__init__(model, optimizer, prune_rate, prune_rate_decay, args, train_loader, slist, Itertrain)
    
        # initialize the gradient momentum buffer
        self.init_buffer()
        self.gmomentum = gmomentum
        self.final_tau = 0.999
        self.curr_gm = gmomentum

    def update_momentum(self, cur_step: int, max_steps: int):
        cur_m = (
            self.final_tau - (self.final_tau - self.gmomentum) * (math.cos(math.pi * cur_step / max_steps) + 1) / 2
        )
        if cur_step >= max_steps:
            cur_m = self.final_tau
        return cur_m

    def init_buffer(self):
        # generate a dummy dictionary as a buffer
        state_copy = self.model.state_dict()
        buffer = OrderedDict()
        
        # initialize the buffer as zero
        for n, v in state_copy.items():
            if 'mask0' in n:
                name = n.replace('.mask0', '')
                buffer[name] = torch.zeros_like(v)
        
        self.buffer_list = []
        for s in self.slist:
            self.buffer_list.append(buffer)
    
    def switch(self, n):
        super().switch(n)
        self.buffer = self.buffer_list[n]
    
    def get_gradient_for_weights(self, weight, name):
        g = super().get_gradient_for_weights(weight, name)
        self.curr_gm = self.update_momentum(self.curr_prune_iter, self.total_prune_iter)

        running_g = self.buffer[name]
        running_g = running_g.mul(self.curr_gm) + (1-self.curr_gm) * g
        self.buffer[name] = running_g
        return running_g
