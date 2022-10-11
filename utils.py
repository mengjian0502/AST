import numpy as np
import pandas as pd
import time
import argparse
import shutil
import tabulate
import torch
import torch.nn.functional as F

torch.manual_seed(0)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    torch.save(state, save_path+filename)
    if is_best:
        shutil.copyfile(save_path+filename, save_path+'model_best.pth.tar')

def print_table(values, columns, epoch, logger):
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    logger.info(table)

def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600*need_hour) / 60)
    need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
    return need_hour, need_mins, need_secs

def log2df(log_file_name):
    '''
    return a pandas dataframe from a log file
    '''
    with open(log_file_name, 'r') as f:
        lines = f.readlines() 
    # search backward to find table header
    num_lines = len(lines)
    for i in range(num_lines):
        if lines[num_lines-1-i].startswith('---'):
            break
    header_line = lines[num_lines-2-i]
    num_epochs = i
    columns = header_line.split()
    df = pd.DataFrame(columns=columns)
    for i in range(num_epochs):
        df.loc[i] = [float(x) for x in lines[num_lines-num_epochs+i].split()]
    return df 


def get_meters(slist):
    """util function for meters"""
    def get_single_meter():
        meters = {}
        meters['loss'] = AverageMeter()
        meters['top1_acc'] = AverageMeter()
        meters['top5_acc'] = AverageMeter() 
        meters['sparsity'] = AverageMeter() 
        return meters

    meters = {}
    for s in slist:
        meters[str(s)] = get_single_meter()
    return meters

def flush_meters(meters, method='avg'):
    results = {}
    assert isinstance(meters, dict), "meters should be dict"
    for name, v in meters.items():
        if name == "sparsity":
            method = "val"
        else:
            method = "avg"
            
        if not isinstance(v, AverageMeter):
            continue 
        if method == 'avg':
            value = v.avg
        elif method == "sum":
            value = v.sum
        elif method == "val":
            value = v.val
        else:
            raise NotImplementedError(f"{method} not implemented yet")
        results[name] = value
    return results

if __name__ == "__main__":
    log = log2df("./save/cifar10/iterative/uresnet32_init1_0.1tofinal0.05_pr0.9_3subnets/uresnet32_lr0.1_wd0.0005_train.log")
    log.to_csv('./save/cifar10/iterative/uresnet32_init1_0.1tofinal0.05_pr0.9_3subnets/uresnet32_lr0.1_wd0.0005_train.csv', index=False)
