"""
Gradient projection with momentum gradient projection
"""


import math
import torch
import torch.nn as nn
from typing import List
from collections import OrderedDict
from .sparse import Mask
from models import SparsConv2d, SparsLinear
import copy


class MomentumGCMask(Mask):
    def __init__(self, model: nn.Module, optimizer=None, prune_rate=0.3, prune_rate_decay=None, args=None, train_loader=None, 
            slist:List=None, Itertrain=False, gmomentum:float=0.99, offset=60):
        super().__init__(model, optimizer, prune_rate, prune_rate_decay, args, train_loader, slist, Itertrain)

        self.offset = offset
        self.gmomentum = gmomentum
        # flag of MA regrow
        self.moving_grad = False

        self.init_buffer()

        # original instant grad
        self.gorg = OrderedDict()

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
        print("[Debug] Buffer reset!")

    def switch(self, n):
        super().switch(n)
        self.buffer = self.buffer_list[n]

    def reg_pre_grads(self,model,n):
        for name, module in model.named_modules():
            if isinstance(module, (SparsConv2d, SparsLinear)):
                pre_grad = copy.deepcopy(module.weight.grad.data)
                # running grad
                running_g = self.buffer[name]
                running_g = running_g.mul(self.gmomentum) + (1-self.gmomentum)*pre_grad
                self.buffer[name] = running_g
                self.buffer_list[n] = self.buffer
    
    def pc_grads(self, model, n):
        for name, module in model.named_modules():
            if isinstance(module, (SparsConv2d, SparsLinear)):
                # running grad
                curr_running_g = self.buffer[name] # running gradient of the current subnet
                
                # running grad of previous 
                idx = n - 1
                ref = idx if idx > 0 else len(self.slist) - 1
                ref_buffer = self.buffer_list[ref]
                ref_running_g = ref_buffer[name]

                g_i_g_j = torch.dot(curr_running_g.flatten(), ref_running_g.flatten())   

                # save the original gradient
                self.gorg[name] = module.weight.grad.data

                if g_i_g_j < 0: 
                    module.weight.grad.data.sub_((g_i_g_j) * ref_running_g / (ref_running_g.norm()**2)) 
    
    def gradient_growth(self, new_mask, total_regrowth, weight, name):
        grad = self.gorg[name]

        # only grow the weights within the current sparsity range
        grad = grad*(new_mask==0).float()
        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        return new_mask

    def step(self, n, bidx, model, regrow=True):
        # update the model
        self.model = model
        
        self.prune_rate_decay.step()
        self.prune_rate = self.prune_rate_decay.get_dr()
        self.steps[n] += 1

        offset = n * self.offset
        # offset = 0

        if self.iter:
            prune_step = self.steps[n] * len(self.slist) if bidx < len(self.slist) else ((self.steps[n]-offset)-1) * len(self.slist)
        else:
            prune_step = self.steps[n]
        # print("prune step={}, n={}, {}".format(prune_step, n, prune_step % self.prune_every_k_steps))
    
        if prune_step >= (self.args.init_prune_epoch * len(self.loader)*self.args.multiplier) and prune_step % self.prune_every_k_steps == 0:
            if prune_step != 0:
                self.pruning(prune_step, bidx)
                if regrow:
                    self.prune_and_regrow(bidx)
                    # self.init_buffer()