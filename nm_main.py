"""
AST: Alternating Sparse Training

@author: Jian Meng, Li Yang
"""

import argparse
import torch
import torch.nn as nn
import os
import logging
import models
from collections import OrderedDict
from torch import optim
from get_data import get_loader
from utils import *
from pruners import CosineDecay, Mask, MaskStat, NM_Mask
from train import train, test, trainIter

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/ImageNet Low precision training')
parser.add_argument('--model', type=str, help='model type')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 64)')

parser.add_argument('--schedule', type=int, nargs='+', default=[60, 120],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--lr_decay', type=str, default='step', help='mode for learning rate decay')
parser.add_argument('--print_freq', default=1, type=int,
                    metavar='N', help='print frequency (default: 200)')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--log_file', type=str, default=None,
                    help='path to log file')
parser.add_argument('--multiplier', type=int, default=1, metavar='N', help='extend training time by multiplier times')                    

# dataset
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset: CIFAR10 / ImageNet_1k')
parser.add_argument('--data_path', type=str, default='./data/', help='data directory')

# model saving
parser.add_argument('--save_path', type=str, default='./save/', help='Folder to save checkpoints and log.')
parser.add_argument('--evaluate', action='store_true', help='evaluate the model')

# resume and fine tune
parser.add_argument('--fine_tune', dest='fine_tune', action='store_true', help='fine tuning from the pre-trained model, force the start epoch be zero')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='full pre-trained model')
parser.add_argument('--resume', default='', type=str, help='path of the pretrained model')

# mixed precision training
parser.add_argument('--ngpu', default=1, type=int, metavar='N', help='number of gpus for parallelism')
parser.add_argument("--mixed_prec", type=str2bool, nargs='?', const=True, default=False, help="enable the mixed precision training or not")

# sparsification
parser.add_argument('--iter', dest='iter', action='store_true', help='iterative pruning scheme')
parser.add_argument('--init-prune-epoch', type=int, default=0, help='The pruning rate / death rate.')
parser.add_argument('--final-prune-epoch', type=int, default=110, help='The density of the overall sparse network.')
parser.add_argument('--prune-rate', type=float, default=0.50, help='The pruning rate / death rate for Zero-Cost Neuroregeneration.')
parser.add_argument('--init-density', type=float, default=0.50, help='Initial pruning rate')
parser.add_argument('--final-density', type=float, default=1.0, help='The density of the overall sparse network.')
parser.add_argument('--update-frequency', type=int, default=1000, metavar='N', help='how many iterations to train between mask update')

# uspars
parser.add_argument('--Mlist', type=float, nargs='+', default=[4, 4, 8, 16], help='targeted granularity')
parser.add_argument('--Nlist', type=float, nargs='+', default=[2, 3, 7, 15], help='targeted granularity')

# pruner type
parser.add_argument('--pruner', type=str, help='pruning method')

# pc_grad
parser.add_argument('--pc_grad', dest='pc_grad', action='store_true', help='gradient projection to sub-nets')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
args.use_cuda = torch.cuda.is_available()  # check GPU

def main():
    # logging
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    
    logger = logging.getLogger('training')
    if args.log_file is not None:
        fileHandler = logging.FileHandler(args.save_path+args.log_file)
        fileHandler.setLevel(0)
        logger.addHandler(fileHandler)
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(0)
    logger.addHandler(streamHandler)
    logger.root.setLevel(0)
    logger.info(args)

    # prepare the dataset
    trainloader, testloader, num_classes = get_loader(args)

    # get the sparsity list for trainer
    args.slist = [n/m for (n,m) in zip(args.Nlist, args.Mlist)]

    # Prepare the model
    logger.info('==> Building model..\n')
    model_cfg = getattr(models, args.model)
    model_cfg.kwargs.update({"num_classes": num_classes, "nspars": len(args.slist), "uspars":True})
    model = model_cfg.base(*model_cfg.args, **model_cfg.kwargs) 
    logger.info(model)

    # resume and fine-tune
    if args.fine_tune:
        checkpoint = torch.load(args.resume)
        checkpoint = checkpoint['state_dict']
        
        new_state_dict = OrderedDict()
        logger.info("=> loading checkpoint...")
        
        for k, v in checkpoint.items():
            name = k
            new_state_dict[name] = v
        
        state_tmp = model.state_dict()
        state_tmp.update(new_state_dict)

        model.load_state_dict(state_tmp)
        logger.info("=> loaded checkpoint!")
    
    if args.use_cuda:
        model = model.cuda()
    
    # loss
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # scaler
    if args.mixed_prec:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs / 2) * args.multiplier, int(args.epochs * 3 / 4) * args.multiplier], last_epoch=-1)

    # initialize pruner
    pr_decay = CosineDecay(args.prune_rate, T_max=len(trainloader)*args.epochs)
    mask = NM_Mask(model, optimizer=optimizer, prune_rate=args.prune_rate, prune_rate_decay=pr_decay, args=args, 
                        train_loader=trainloader, slist=args.slist, Itertrain=True, NList=args.Nlist, MList=args.Mlist)
    
    # Evaluate
    if args.evaluate:
        logger.info("Evaluate!")
        model.switch(0)
        test_res = test(model, testloader, criterion, args, slist=args.slist)
        
        logger.info('Test accuracy:')
        for s in args.slist:
            logger.info("Test accuracy of {} model = {:.2f}".format(s, test_res[str(s)]['top1_acc']))
        
        mask.reg_masks(train=False)
        total_params, spars_params = mask._param_stats()
        print("sparsity = {:.3f}".format(spars_params / total_params*100))

        meter = MaskStat(model, args.slist, logger, M=16)
        meter.sparsity()
        exit()

    mask.reg_masks(train=True)

    # training
    best_acc = 0.
    start_time = time.time()
    epoch_time = AverageMeter()
    total_iter = AverageMeter()
    
    columns = ['ep', 'lr']
    for s in args.slist:
        columns.append(f"tra_acc{s}")
        columns.append(f"te_acc{s}")
        columns.append(f"spars_{s}")
    columns.append('best_acc')
    
    # meter = MaskStat(model, args.slist, logger)
    for epoch in range(args.epochs):
        # training phase
        if not args.iter:
            train_res = train(model, trainloader, criterion, optimizer, scaler, mask, args, slist=args.slist)
        else:
            train_res = trainIter(model, trainloader, criterion, optimizer, scaler, mask, args, slist=args.slist, total_iter=total_iter)

        lr_scheduler.step()
        
        # test phase
        test_res = test(model, testloader, criterion, args, slist=args.slist)

        # best flag
        is_best = test_res[str(args.slist[-1])]["top1_acc"] > best_acc
        
        if is_best:
            best_acc = test_res[str(args.slist[-1])]["top1_acc"]

        state = {
            'state_dict': model.state_dict(),
            'acc': best_acc,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
        }

        filename='checkpoint.pth.tar'
        save_checkpoint(state, is_best, args.save_path, filename=filename)
        
        # logname = f"./layerwise_overlap_epoch{epoch}.csv"
        # meter.overlap(args.save_path+logname)

        # record time
        e_time = time.time() - start_time
        epoch_time.update(e_time)
        start_time = time.time()

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        values = [epoch + 1, optimizer.param_groups[0]['lr']]

        for s in args.slist:
            values.append(train_res[str(s)]['top1_acc'])
            values.append(test_res[str(s)]['top1_acc'])
            values.append(train_res[str(s)]['sparsity'])
        values.append(best_acc)
        print_table(values, columns, epoch, logger)
        print(need_time)


if __name__ == '__main__':
    main()
