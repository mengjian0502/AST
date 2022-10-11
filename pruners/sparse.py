"""
Masking
"""
import os
import math
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from models import SparsConv2d, SparsLinear
from typing import List
from torch import Tensor
from collections import OrderedDict
import copy
from utils import AverageMeter


class CosineDecay(object):
    def __init__(self, prune_rate, T_max, eta_min=0.005, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=prune_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self):
        return self.sgd.param_groups[0]['lr']


class Mask(object):
    def __init__(self, model:nn.Module, optimizer=None, prune_rate=0.3, prune_rate_decay=None, args=None, train_loader=None, slist:List=None, Itertrain=False, offset=60):
        self.args = args
        self.loader = train_loader
        self.prune_rate_decay = prune_rate_decay
        self.prune_rate = prune_rate   # prune rate during the "prune and regrow" step 
        self.optimizer = optimizer
        self.model = model
        self.offset = offset

        # global mask
        self.masks = {}

        # prunnig
        self.total_params = 0 
        self.prune_every_k_steps = self.args.update_frequency
        self.pruning_count = {}

        self.steps = [0 for i in range(len(slist))]

        # switchable pruning
        self.slist = slist

        # iterative pruning
        self.iter = Itertrain

        # prune rate
        self.curr_prune_rate = 0.

        # pregrad
        self.pre_grads = {}

        # original gradient copy
        self.gorg = OrderedDict()

    def switch(self, n):
        self.final_density = self.slist[n]
        
        # restate the masks for the subnet
        for name, module in self.model.named_modules():
            if isinstance(module, (SparsConv2d, SparsLinear)):
                self.masks[name] = module.mask

    def _param_stats(self):
        total_params = 0
        spars_params = 0
        for n, m in self.model.named_modules():
            if isinstance(m, (SparsConv2d, SparsLinear)):
                mask = m.mask.data
                total_params += mask.numel()
                spars_params += mask[mask.eq(0)].numel()
        return total_params, spars_params

    def _layer_stats(self):
        self.name2nonzeros = {}
        self.name2zeros = {}

        for name, mask in self.masks.items():
            self.name2nonzeros[name] = mask.sum().item()
            self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]
        
    def reg_masks(self, train):
        for name, module in self.model.named_modules():
            if isinstance(module, (SparsConv2d, SparsLinear)):
                self.masks[name] = module.mask
        
        if train:
            # initial sparsity
            self.init(self.args.init_density)
        
            # apply mask
            self.apply_mask(cnt=0)
            # self.apply_mask(cnt=1)

    def reg_pre_grads(self, model, n):
        for name, module in self.model.named_modules():
            if isinstance(module, (SparsConv2d, SparsLinear)):
                self.pre_grads[name] = copy.deepcopy(module.weight.grad.data)
    
    def pc_grads(self, model, n):
        for name, module in model.named_modules():
            if isinstance(module, (SparsConv2d, SparsLinear)):

                pre_grad = self.pre_grads[name]
                g_i_g_j = torch.dot(module.weight.grad.data.flatten(), pre_grad.flatten())

                # save the original gradient
                self.gorg[name] = module.weight.grad.data   

                if g_i_g_j < 0: 
                    module.weight.grad.data.sub_((g_i_g_j) * pre_grad / (pre_grad.norm()**2)) 

    def init(self, density:float, erk_power_scale:float=1.0):
        print('initialize by ERK')
        self.total_params, _ = self._param_stats()

        is_epsilon_valid = False
        dense_layers = set()
        while not is_epsilon_valid:
            divisor = 0
            rhs = 0
            raw_probabilities = {}
            for name, mask in self.masks.items():
                n_param = mask.numel()
                n_zeros = n_param * (1 - density)
                n_ones = n_param * density

                if name in dense_layers:
                    # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                    rhs -= n_zeros

                else:
                    # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                    # equation above.
                    rhs += n_ones
                    # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                    raw_probabilities[name] = (np.sum(list(mask.size())) / mask.numel()) ** erk_power_scale
                    # Note that raw_probabilities[mask] * n_param gives the individual
                    # elements of the divisor.
                    divisor += raw_probabilities[name] * n_param
            # By multipliying individual probabilites with epsilon, we should get the
            # number of parameters per layer correctly.
            epsilon = rhs / divisor
            # If epsilon * raw_probabilities[mask.name] > 1. We set the sparsities of that
            # mask to 0., so they become part of dense_layers sets.
            max_prob = np.max(list(raw_probabilities.values()))
            max_prob_one = max_prob * epsilon
            if max_prob_one > 1:
                is_epsilon_valid = False
                for mask_name, mask_raw_prob in raw_probabilities.items():
                    if mask_raw_prob == max_prob:
                        print(f"Sparsity of var:{mask_name} had to be set to 0.")
                        dense_layers.add(mask_name)
            else:
                is_epsilon_valid = True

        density_dict = {}
        total_nonzero = 0.0

        # With the valid epsilon, we can set sparsities of the remaning layers.
        for name, mask in self.masks.items():
            n_param = mask.numel()
            if name in dense_layers:
                density_dict[name] = 1.0
            else:
                probability_one = epsilon * raw_probabilities[name]
                density_dict[name] = probability_one
            print(
                f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}"
            )
            self.masks[name][:] = (torch.rand(mask.shape) < density_dict[name]).float().data.cuda()

            total_nonzero += density_dict[name] * mask.numel()
        print(f"Overall sparsity {1 - total_nonzero / self.total_params}")


    def apply_mask(self, cnt):
        for n, m in self.model.named_modules():
            if isinstance(m, (SparsConv2d, SparsLinear)):
                m.mask.data = self.masks[n].clone()
                if cnt == 0:
                    m.mask0.data = self.masks[n].clone()
                elif cnt == 1:
                    m.mask1.data = self.masks[n].clone()

    def collect_score(self):
        weight_abs = []
        for n, m in self.model.named_modules():
            if isinstance(m, (SparsConv2d, SparsLinear)):
                weight_abs.append(m.weight.data.abs())
        mp_scores = torch.cat([torch.flatten(x) for x in weight_abs])
        return mp_scores

    def update_mask(self, threshold):
        for n, m in self.model.named_modules():
            if isinstance(m, (SparsConv2d, SparsLinear)):
                self.masks[n] = m.weight.abs().gt(threshold).float()

    def pruning(self, step, bidx, cnt):
        """
        Step 1: Scheduled pruning
        """
        self.curr_prune_iter = int(step / self.prune_every_k_steps)
        final_iter = int((self.args.final_prune_epoch * len(self.loader)*self.args.multiplier) / self.prune_every_k_steps)
        ini_iter = int((self.args.init_prune_epoch * len(self.loader)*self.args.multiplier) / self.prune_every_k_steps)
        self.total_prune_iter = final_iter - ini_iter

        # message
        print(f'Pruning Progress is {self.curr_prune_iter - ini_iter} / {self.total_prune_iter}')
        if self.curr_prune_iter >= ini_iter and self.curr_prune_iter <= final_iter:
            # update sparsity schedule
            ramping_decay = (1 - ((self.curr_prune_iter - ini_iter) / self.total_prune_iter)) ** 3
            self.curr_prune_rate = (1 - self.args.init_density) + (self.args.init_density - self.final_density) * (1 - ramping_decay)

            # magnitude score
            mp_scores = self.collect_score()
            num_params_to_keep = int(len(mp_scores) * (1 - self.curr_prune_rate))
            topkscore, _ = torch.topk(mp_scores, num_params_to_keep, sorted=True)
            threshold = topkscore[-1]

            # update and apply the masks
            self.update_mask(threshold)
            self.apply_mask(cnt)
        else:
            self.curr_prune_rate = 0.
        
        # sparsity
        total_params, spars_params = self._param_stats()
        sparsity = spars_params / total_params
        print("Sparsity after pruning at step [{}] = {:.3f}".format(bidx, sparsity*100))

    def overlap(self, new_mask, mask_ref):
        num_all = mask_ref.numel()
        num_spars = num_all - mask_ref.sum()
        
        new_mask = new_mask.float()
        diff = new_mask[mask_ref.eq(0.)]
        s = diff.sum() / num_spars
        return s.item()

    def prune_and_regrow(self, step, cnt):
        """
        Step 2: Layer-wise pruning followed by re-growing
        """

        # layer statistics
        self._layer_stats()
        
        overlap_all = {}
        # prune
        for n, m in self.model.named_modules():
            if isinstance(m, (SparsConv2d, SparsLinear)):
                weight = m.weight
                
                # update mask for pruning
                new_mask = self.magnitude_death(weight, n)
                self.pruning_count[n] = int(self.name2nonzeros[n] - new_mask.sum().item())

                # regrow
                new_mask = self.gradient_growth(new_mask, self.pruning_count[n], weight, n)
                
                # how many elements are re-grown based on the gradient
                old_mask = self.masks[n]
                s = self.overlap(new_mask, old_mask)
                overlap_all[n] = s

                # apply mask
                m.mask.data = new_mask.float().clone()
                
                if cnt == 0:
                    m.mask0.data = new_mask.float().clone()
                elif cnt == 1:
                    m.mask1.data = new_mask.float().clone()
                
                # record mask
                self.masks[n] = new_mask

        # sparsity
        total_params, spars_params = self._param_stats()
        sparsity = spars_params / total_params
        
        # # overlap
        # filename = f"ngrow_at_step{step}.csv"
        # save_path = os.path.join(self.args.save_path, filename)
        # odf = pd.DataFrame(overlap_all, index=[0])
        # odf = odf.T
        # odf.to_csv(save_path)
        print("Sparsity after regrow at step [{}] = {:3f}".format(step, sparsity*100))

    def magnitude_death(self, weight, name):
        """
        Step 2-1: Remove the most non-significant weights inside remaining weights
        """
        num_remove = math.ceil(self.prune_rate*self.name2nonzeros[name])
        if num_remove == 0.0: return weight.data != 0.0
        num_zeros = self.name2zeros[name]
        k = math.ceil(num_zeros + num_remove)
        x, idx = torch.sort(torch.abs(weight.data.view(-1)))
        threshold = x[k-1].item()
        return (torch.abs(weight.data) > threshold)
    
    def gradient_growth(self, new_mask, total_regrowth, weight, name):
        """
        Step 2-2: Regrow the weights with the most significant gradient
        """
        if self.args.pc_grad:
            grad = self.gorg[name]
        else:
            grad = self.get_gradient_for_weights(weight, name)
        
        # only grow the weights within the current sparsity range
        grad = grad*(new_mask==0).float()
        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        return new_mask

    def get_gradient_for_weights(self, weight, name):
        grad = weight.grad.clone()
        return grad

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
                self.pruning(prune_step, bidx, n)
                if regrow:
                    self.prune_and_regrow(bidx, n)
        
    def nlprune(self, tspars:float, cnt:int):
        r"""Direct pruning without learning (after training)
        """
        mp_scores = self.collect_score()
        num_params_to_keep = int(len(mp_scores) * (1 - tspars))
        
        topkscore, _ = torch.topk(mp_scores, num_params_to_keep, sorted=True)
        threshold = topkscore[-1]

        # update and apply the masks
        self.update_mask(threshold)
        self.apply_mask(cnt)

class MaskStat:
    def __init__(self, model:nn.Module, slist:List, logger, M=None):
        self.model = model
        self.slist = slist
        self.logger = logger
        self.M = M

    def _get_groups(self, tensor:Tensor):
        length = tensor.numel()
        group = int(length/self.M)
        return group
    
    def sparsity(self):
        sparsity_all = OrderedDict()
        cnt = 0
        total_param = 0
        total_nz = 0
        for n, m in self.model.named_modules():
            if isinstance(m, (SparsConv2d, SparsLinear)):
                spars_list = []
                for i in range(len(self.slist)):
                    mask = m.__getattr__(f"mask{i}")
                    
                    # for structured mask pattern
                    if self.M is not None:
                        group = self._get_groups(mask)
                        if len(mask.size()) == 4:
                            m = mask.permute(0,2,3,1).reshape(group, int(self.M))
                        elif len(mask.size()) == 2:
                            m = mask.reshape(group, int(self.M))
                        
                        msum = torch.sum(m, dim=1)
                        print("Unique sparse pattern of layer {} = {}".format(n, msum.unique().cpu().numpy()))

                    nz = mask.sum().item()
                    
                    total_param += mask.numel()
                    total_nz += nz

                    spars = 1 - nz/mask.numel()
                    spars_list.append(spars)
                name = n +'.sparsity'
                sparsity_all[name] = np.array(spars_list)
                cnt += 1
        self.sdf = pd.DataFrame.from_dict(sparsity_all)
        self.sdf = self.sdf.T
        self.sdf.to_csv("./layerwise_sparsity.csv")
        overall_sparsity = 1 - total_nz / total_param
        print(overall_sparsity)
    
    def overlap(self, save_path):
        overlap_all = OrderedDict()
        cnt = 0
        for n, m in self.model.named_modules():
            if isinstance(m, (SparsConv2d, SparsLinear)):
                mask_ref = m.__getattr__(f"mask{0}")
                num_all = mask_ref.numel()
                overlap = []
                for i in range(1, len(self.slist)):
                    mask = m.__getattr__(f"mask{i}")
                    xor = torch.bitwise_xor(mask.int(), mask_ref.int())
                    s = xor[xor.eq(0.)].numel() / num_all
                    overlap.append(s)
                name = n +'.overlap'
                overlap_all[name] = np.array(overlap)
                cnt += 1
        self.odf = pd.DataFrame.from_dict(overlap_all)
        self.odf = self.odf.T
        # self.odf.to_csv("./layerwise_overlap_Iter.csv")
        self.odf.to_csv(save_path)