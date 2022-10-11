"""
Training and testing
"""

import time
import torch
import torch.nn as nn
from models.sparsmodule import SparsConv2d, SparsLinear
from utils import AverageMeter, accuracy, get_meters, flush_meters

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def train(model, trainloader=None, criterion=None, optimizer=None, scaler=None, mask=None, args=None, slist=None, total_iter=None):
    meters = get_meters(slist)

    # switch to train mode
    model = model.train()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        targets = targets.cuda(non_blocking=True)
        inputs = inputs.cuda()

        # switchable prune and grow
        optimizer.zero_grad()
        for i, s in enumerate(slist):
            meter = meters[str(s)]

            model.switch(i)     # switch the model
            mask.switch(i)      # switch the mask
            
            if args.mixed_prec:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    
                    # make sure the output data precision is float16
                    assert outputs.dtype is torch.float16
                    loss = criterion(outputs, targets)

                    # compute gradient            
                    scaler.scale(loss).backward()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # compute gradient
                loss.backward()

            # back prop 
            if args.mixed_prec:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # prune and regrow of the current subnet
            mask.step(i, model, total_iter.val)

            # sparse gradient
            for module in model.modules():
                if isinstance(module, (SparsConv2d, SparsLinear)):
                    if module.weight.grad is not None:
                        module.weight.grad.data.mul_(module.mask)

            # accuracy of the subnet
            prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))

            if batch_idx == 0:
                print("[Debug]: Targeted density switch to = {}".format(s))

            # sparsity
            total_param, spars_param = mask._param_stats()
            sparsity = spars_param / total_param*100
            
            if batch_idx == 390:
                print("Density={}; sparsity={}".format(s, sparsity))

            # record
            meter['loss'].update(loss.item(), inputs.size(0))
            meter['top1_acc'].update(prec1.item(), inputs.size(0))
            meter['top5_acc'].update(prec5.item(), inputs.size(0))
            meter['sparsity'].update(sparsity, 1)

            total_iter.val += 1

    results = {}
    for s in slist:
        res = flush_meters(meters[str(s)])
        results[str(s)] = res 
    return results


def test(model, testloader=None, criterion=None, args=None, slist=None):
    meters = get_meters(slist)

    model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            targets = targets.long().cuda(non_blocking=True)
            inputs = inputs.float().cuda()

            # switch
            for i, s in enumerate(slist):
                meter = meters[str(s)]
                
                # switch the model
                if args.ngpu > 1:
                    model.module.switch(i)
                else:
                    model.switch(i)

                if args.mixed_prec:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                    
                        # make sure the output data precision is float16
                        assert outputs.dtype is torch.float16
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                # compute accuracy
                prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
            
                # record
                meter['loss'].update(loss.item(), inputs.size(0))
                meter['top1_acc'].update(prec1.item(), inputs.size(0))
                meter['top5_acc'].update(prec5.item(), inputs.size(0))
    
    
    results = {}
    for s in slist:
        res = flush_meters(meters[str(s)])
        results[str(s)] = res 
    return results

def trainIter(model, trainloader=None, criterion=None, optimizer=None, scaler=None, mask=None, args=None, slist=None, total_iter=None):
    meters = get_meters(slist)

    # switch to train mode
    model = model.train()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        targets = targets.long().cuda(non_blocking=True)
        inputs = inputs.float().cuda()

        cnt = total_iter.val % len(slist)
        
        if batch_idx < len(slist):
            print("[Debug]: [{}]/[{}] Targeted density switch to {} = {}".format(total_iter.val, len(trainloader), cnt, slist[cnt]))
        
        # switchable prune and grow
        optimizer.zero_grad()

        meter = meters[str(slist[cnt])]
        
        if args.ngpu > 1:
            model.module.switch(cnt)
        else:
            model.switch(cnt)
            
        mask.switch(cnt)

        if args.mixed_prec:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                
                # make sure the output data precision is float16
                assert outputs.dtype is torch.float16
                loss = criterion(outputs, targets)

                # compute gradient            
                scaler.scale(loss).backward()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # compute gradient
            loss.backward()

        # latch the current mask
        if args.pruner == "mono-regrow-subset":
            if total_iter.val >= args.update_frequency:
                mask.latch(cnt)

        # prune and regrow of the current subnet
        mask.step(cnt, total_iter.val, model)

        # sparse gradient
        for module in model.modules():
            if isinstance(module, (SparsConv2d, SparsLinear)):
                if module.weight.grad is not None:
                    module.weight.grad.data.mul_(module.mask)

        #  gradient projection
        if args.pc_grad:
            mask.reg_pre_grads(model, cnt)
            if total_iter.val != 0:
                mask.pc_grads(model, cnt)


        # accuracy of the subnet
        prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))

        # sparsity
        total_param, spars_param = mask._param_stats()
        sparsity = spars_param / total_param*100
        
        # record
        meter['loss'].update(loss.item(), inputs.size(0))
        meter['top1_acc'].update(prec1.item(), inputs.size(0))
        meter['top5_acc'].update(prec5.item(), inputs.size(0))
        meter['sparsity'].update(sparsity, 1)


        # back prop 
        if args.mixed_prec:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        total_iter.val += 1
    
    results = {}
    for s in slist:
        res = flush_meters(meters[str(s)])
        results[str(s)] = res 
    return results
