"""
Structured Masking
"""
import math
import torch
import torch.nn as nn
from torch import Tensor
from models import SparsConv2d, SparsLinear
from typing import List
from .sparse import Mask

class NM_Mask(Mask):
    def __init__(self, model:nn.Module, optimizer=None, prune_rate=0.3, prune_rate_decay=None, args=None, train_loader=None, slist:List=None, Itertrain=False, NList:List=None, MList:List=None):
        super(NM_Mask, self).__init__(model, optimizer, prune_rate, prune_rate_decay, args, train_loader, slist, Itertrain)

        # N:M sparsity
        self.NList = NList
        self.MList = MList

        # probability ramping
        self.final_prob = 0.0

    def switch(self, n):
        self.N = self.NList[n]
        self.M = self.MList[n]

        # restate the masks for the subnet
        for name, module in self.model.named_modules():
            if isinstance(module, (SparsConv2d, SparsLinear)):
                self.masks[name] = module.mask

    def _get_groups(self, tensor:Tensor):
        length = tensor.numel()
        group = int(length/self.M)
        return group

    def update_mask(self, prob):
        for n, m in self.model.named_modules():
            if isinstance(m, ((SparsConv2d, SparsLinear))):
                weight = m.weight.clone()
                group = self._get_groups(weight)

                if isinstance(m, SparsConv2d):
                    weight_temp = weight.detach().abs().permute(0,2,3,1).reshape(group, int(self.M))
                elif isinstance(m, SparsLinear):
                    weight_temp = weight.detach().abs().reshape(group, int(self.M))

                index = torch.argsort(weight_temp, dim=1)[:, :int(self.M-self.N)]
                w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
                w_b = w_b.scatter_(dim=1, index=index, value=0)
                
                # probability sample 
                if prob < 1.0:
                    rnumel = math.ceil((1-prob)*group)
                    ridx = torch.randperm(rnumel)
                    w_b[ridx] = 1.
                
                # reshape
                if isinstance(m, SparsConv2d):
                    w_b = w_b.reshape(weight.permute(0,2,3,1).shape)
                    w_b = w_b.permute(0,3,1,2)
                elif isinstance(m, SparsLinear):
                    w_b = w_b.reshape(weight.shape)

                # update the mask
                self.masks[n] = w_b

    
    def structured_layer_stats(self):
        self.name2nzgrp = {}
        self.name2zerogrp = {}

        for name, mask in self.masks.items():
            group = self._get_groups(mask)
            
            if len(mask.size()) == 4:
                m = mask.permute(0,2,3,1).reshape(group, int(self.M))
            elif len(mask.size()) == 2:
                m = mask.reshape(group, int(self.M))
            
            gsum = m.sum(dim=1)
            nzgrp = gsum[gsum.eq(self.M)].numel()

            # sparse and non sparse groups
            self.name2nzgrp[name] = nzgrp
            self.name2zerogrp[name] = group - nzgrp


    def pruning(self, step, bidx):
        """
        Step 1: Scheduled N:M pruning
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
            
            # current probability
            self.curr_prob = (1 - self.args.init_density) + (self.args.init_density - self.final_prob) * (1 - ramping_decay)

            # update mask
            self.update_mask(self.curr_prob)
            self.apply_mask()
        
        # sparsity stats
        total_params, spars_params = self._param_stats()
        sparsity = spars_params / total_params
        
        # print("Sparsity after pruning at step [{}] = {:3f}".format(bidx, sparsity*100))
        print("Sparsity after pruning at step [{}] = {:.3f} with prob={:.3f}".format(bidx, sparsity*100, self.curr_prob))

    def prune_and_regrow(self, step):
        
        # layer statistics 
        self.structured_layer_stats()
        self._layer_stats()
        # prune
        for n, m in self.model.named_modules():
            if isinstance(m, (SparsConv2d, SparsLinear)):
                weight = m.weight
                
                # update the mask for pruning
                new_mask, num_remove = self.group_death(weight, n, m)
                self.pruning_count[n] = num_remove
                # self.pruning_count[n] = int(self.name2nonzeros[n] - new_mask.sum().item())

                # regrow
                new_mask, remained = self.grp_grad_growth(new_mask, self.pruning_count[n], weight)

                # apply mask
                if num_remove > 0:
                    m.mask = new_mask.clone()
                    # record mask
                    self.masks[n] = new_mask
        
        # sparsity
        total_params, spars_params = self._param_stats()
        sparsity = spars_params / total_params
        print("Overall sparsity after regrow at step [{}] = {:.3f}".format(step, sparsity*100))

    
    def group_death(self, weight, name, module):
        group = self._get_groups(weight)

        # number of groups we want to remove (temporarily)
        num_remove = int(self.prune_rate*self.name2nzgrp[name]) 
        
        if num_remove == 0.0: 
            w_b = weight.data.ne(0).float()
        else:
            # number of pruned groups
            num_zeros = self.name2zerogrp[name]
            
            # total number of sparse groups
            k = int(num_zeros + num_remove)

            if isinstance(module, SparsConv2d):
                weight_temp = weight.detach().abs().permute(0,2,3,1).reshape(group, int(self.M))
            elif isinstance(module, SparsLinear):
                weight_temp = weight.detach().abs().reshape(group, int(self.M))

            index = torch.argsort(weight_temp, dim=1)[:, :int(self.M-self.N)]

            w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
            w_b = w_b.scatter_(dim=1, index=index, value=0) # fill all the groups with N:M sparsity

            # prune more unimportant groups
            wgsum = torch.sum(weight_temp.abs(), dim=1)
            y, idx = torch.sort(torch.abs(wgsum).flatten())
            w_b[idx[:(wgsum.size(0)-k)]] = 1.
            
            # reshape
            if isinstance(module, SparsConv2d):
                w_b = w_b.reshape(weight.permute(0,2,3,1).shape)
                w_b = w_b.permute(0,3,1,2)
            elif isinstance(module, SparsLinear):
                w_b = w_b.reshape(weight.shape)
        return w_b, num_remove

    def grp_grad_growth(self, new_mask, total_regrowth, weight):
        grad = self.get_gradient_for_weights(weight)
        group = self._get_groups(grad)
        
        # gradient group
        if len(new_mask.size())==4:
            gradgrp = grad.abs().permute(0,2,3,1).reshape(group, int(self.M))
            m = new_mask.permute(0,2,3,1).reshape(group, int(self.M))
        elif len(new_mask.size())==2:
            gradgrp = grad.abs().reshape(group, int(self.M))
            m = new_mask.reshape(group, int(self.M))

        # only grow the weights within the current sparsity
        msum = torch.sum(m, dim=1)
        sidx = msum.eq(self.N).float()

        gradgrp = gradgrp*sidx[:, None]
        gsum = torch.sum(gradgrp, dim=1)
        y, idx = torch.sort(gsum.flatten(), descending=True)            
        
        # regrow
        m[idx[:total_regrowth]] = 1.0
        msum = torch.sum(m, dim=1)
        # print(msum.unique())
        
        # reshape
        if len(new_mask.size())==4:
            rgmask = m.reshape(new_mask.permute(0,2,3,1).shape)
            rgmask = rgmask.permute(0,3,1,2)
        elif len(new_mask.size())==2:
            rgmask = m.reshape(new_mask.shape)

        return rgmask, msum[msum.eq(self.N)].numel()