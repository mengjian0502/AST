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
import wandb
from collections import OrderedDict
from torch import optim
from get_data import get_loader
from utils import *
from pruners import CosineDecay, Mask, MaskStat, MultiRegMask, MonoRegMask, MonoRegSubsetMask, MomentumMask, CyclicMomentumMask, MomentumGCMask, EnsembleMask
from train import LabelSmoothing, train, test, trainIter
from imagenet_loaders import get_train_loader, get_val_loader
from cosineLR_warmup import LinearWarmupCosineAnnealingLR

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
parser.add_argument('--lr_scheduler', type=str, help='learning rate scheduler')
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
parser.add_argument('--resume', default=None, type=str, help='path of the pretrained model')

# mixed precision training
parser.add_argument("--mixed_prec", type=str2bool, nargs='?', const=True, default=False, help="enable the mixed precision training or not")

# data parallel
parser.add_argument('--ngpu', default=1, type=int, metavar='N', help='number of gpus for parallelism')

# sparsification
parser.add_argument('--iter', dest='iter', action='store_true', help='iterative pruning scheme')
parser.add_argument('--init-prune-epoch', type=int, default=0, help='The pruning rate / death rate.')
parser.add_argument('--final-prune-epoch', type=int, default=110, help='The density of the overall sparse network.')
parser.add_argument('--prune-rate', type=float, default=0.50, help='The pruning rate / death rate for Zero-Cost Neuroregeneration.')
parser.add_argument('--init-density', type=float, default=0.50, help='Initial pruning rate')
parser.add_argument('--final-density', type=float, default=0.05, help='The density of the overall sparse network.')
parser.add_argument('--update-frequency', type=int, default=1000, metavar='N', help='how many iterations to train between mask update')
parser.add_argument('--gmomentum', type=float, default=0.99, help='Momentum of the moving gradient')
parser.add_argument('--iteroffset', type=int, default=1000, metavar='N', help='how many iterations between two consecutive subnets')
parser.add_argument("--reset_buffer", type=str2bool, nargs='?', const=True, default=False, help="reset the cyclic momentum grad buffer")

# pc_grad
parser.add_argument('--pc_grad', dest='pc_grad', action='store_true', help='gradient projection to sub-nets')

# pruner type
parser.add_argument('--pruner', type=str, help='pruning method')

# label smoothing
parser.add_argument('--label_smoothing', default=0.0, type=float, help='label smoothing')

# online logging
parser.add_argument("--wandb", type=str2bool, nargs='?', const=True, default=False, help="enable the wandb cloud logger")
parser.add_argument("--name")
parser.add_argument("--project")
parser.add_argument("--entity", default=None, type=str)

# uspars
parser.add_argument('--slist', type=float, nargs='+', default=[0.6, 0.5, 0.25, 0.1], help='density candidates (sparsity = 1-density)')

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
    if 'cifar' in args.dataset:
        trainloader, testloader, num_classes = get_loader(args)
    elif 'imagenet' in args.dataset:
        trainloader = get_train_loader(args.data_path, batch_size=args.batch_size)
        testloader = get_val_loader(args.data_path, batch_size=args.batch_size)
        num_classes = 1000

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
        if args.ngpu > 1:
            model = torch.nn.DataParallel(model)
            logger.info("Data parallel!")
    
    # loss
    if args.label_smoothing > 0:
        criterion = LabelSmoothing(args.label_smoothing).cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # scaler
    if args.mixed_prec:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # lr scheduler
    if args.lr_scheduler == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-10)
        # lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.init_prune_epoch, max_epochs=args.epochs, warmup_start_lr=args.lr/args.init_prune_epoch, eta_min=1e-10)
    else:
        if args.dataset == "imagenet":
            milestones = [30, 60, 90]
            if args.resume is not None:
                milestones = [15, 25]
        else:
            milestones=[int(args.epochs / 2) * args.multiplier, int(args.epochs * 3 / 4) * args.multiplier]
        
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, last_epoch=-1)

    # initialize pruner
    pr_decay = CosineDecay(args.prune_rate, T_max=len(trainloader)*args.epochs)
    
    if args.pruner == "iterative":
        mask = Mask(model, optimizer=optimizer, prune_rate=args.prune_rate, prune_rate_decay=pr_decay, args=args, train_loader=trainloader, slist=args.slist, Itertrain=True, offset=args.iteroffset)
    elif args.pruner == "momentum-iter":
        mask = MomentumMask(model, optimizer=optimizer, prune_rate=args.prune_rate, prune_rate_decay=pr_decay, args=args, train_loader=trainloader, slist=args.slist, Itertrain=True, gmomentum=args.gmomentum)
    elif args.pruner == "multi-regrow":
        mask = MultiRegMask(model, optimizer=optimizer, prune_rate=args.prune_rate, prune_rate_decay=pr_decay, args=args, train_loader=trainloader, slist=args.slist, Itertrain=True)
    elif args.pruner == "mono-regrow":
        mask = MonoRegMask(model, optimizer=optimizer, prune_rate=args.prune_rate, prune_rate_decay=pr_decay, args=args, train_loader=trainloader, slist=args.slist, Itertrain=True)
    elif args.pruner == "mono-regrow-subset":
        mask = MonoRegSubsetMask(model, optimizer=optimizer, prune_rate=args.prune_rate, prune_rate_decay=pr_decay, args=args, train_loader=trainloader, slist=args.slist, Itertrain=True)
    elif args.pruner == "cyclic-iter":
        mask = CyclicMomentumMask(model, optimizer=optimizer, prune_rate=args.prune_rate, prune_rate_decay=pr_decay, args=args, train_loader=trainloader, slist=args.slist, Itertrain=True, gmomentum=args.gmomentum, offset=args.iteroffset)
    elif args.pruner == "momentum_gc":
        mask = MomentumGCMask(model, optimizer=optimizer, prune_rate=args.prune_rate, prune_rate_decay=pr_decay, args=args, train_loader=trainloader, slist=args.slist, Itertrain=True, gmomentum=args.gmomentum, offset=args.iteroffset)
    elif args.pruner == "ensemble":
        mask = EnsembleMask(model, optimizer=optimizer, prune_rate=args.prune_rate, prune_rate_decay=pr_decay, args=args, train_loader=trainloader, slist=args.slist)
    else:
        raise NotImplemented

    logger.info("Pruner type = {}: {}".format(args.pruner, type(mask)))

    # Evaluate
    if args.evaluate:
        logger.info("Evaluate!")
        # f = 0.25
        # subset = 0
        # total = 0
        # for n, m in model.named_modules():
        #     if isinstance(m, nn.Conv2d):
        #         if m.in_channels == 3:
        #             cin = m.in_channels
        #         else:
        #             cin = int(f * m.in_channels)
        #         cout = int(f*m.out_channels)
        #         subset += cout*cin*m.kernel_size[0]*m.kernel_size[0]
        #         total += m.weight.numel()
        #     if isinstance(m, nn.Linear):
        #         total += m.weight.numel()
        #         subset += m.weight.numel()*f
        # sparsity = 1 - subset/total
        # print(sparsity)

        test_res = test(model, testloader, criterion, args, slist=args.slist)
        
        logger.info('Test accuracy:')
        for s in args.slist:
            logger.info("Test accuracy of {} model = {:.2f}".format(s, test_res[str(s)]['top1_acc']))
        
        mask.reg_masks(train=False)
        total_params, spars_params = mask._param_stats()
        print("sparsity = {:.3f}".format(spars_params / total_params*100))

        meter = MaskStat(model, args.slist, logger)
        save_path = os.path.join(args.save_path, "layerwise_overlap_Iter.csv")
        meter.overlap(save_path)
        # meter.sparsity()

        # # Direct pruning without training

        # test_res = test(model, testloader, criterion, args, slist=args.slist)
        # logger.info('Test accuracy:')
        # for s in args.slist:
        #     logger.info("Test accuracy of {} model = {:.2f}".format(s, test_res[str(s)]['top1_acc']))
        exit()

    if args.resume is not None:
        mask.reg_masks(train=False)
        # mask.nlprune(tspars=0.9, cnt=0)
        total_params, spars_params = mask._param_stats()
        print("Resumed sparsity = {:.3f}".format(spars_params / total_params*100))
    else:
        mask.reg_masks(train=True)

    # training
    best_acc = 0.
    start_time = time.time()
    epoch_time = AverageMeter()
    total_iter = AverageMeter()
    
    # online logging
    if args.wandb:
        wandb_logger = wandb.init(entity=args.entity, project=args.project, name=args.name, config={"lr":args.lr})
        # wandb_logger.watch(model, criterion=criterion, log_freq=1)
        wandb_logger.config.update(args)


    meter = MaskStat(model, args.slist, logger)
    for epoch in range(args.epochs):
        # training phase
        if not args.iter:
            train_res = train(model, trainloader, criterion, optimizer, scaler, mask, args, slist=args.slist, total_iter=total_iter)
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
        
        # if len(args.slist) > 1:
        #     logname = f"./layerwise_overlap_epoch{epoch}.csv"
        #     meter.overlap(args.save_path+logname)

        # record time
        e_time = time.time() - start_time
        epoch_time.update(e_time)
        start_time = time.time()

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        logger_dict = {'ep': epoch+1, 'lr':optimizer.param_groups[0]['lr']}

        for s in args.slist:
            logger_dict[f"loss{s}"] = train_res[str(s)]['loss']
            logger_dict[f"tra_acc{s}"] = train_res[str(s)]['top1_acc']
            logger_dict[f"te_acc{s}"] = test_res[str(s)]['top1_acc']
            logger_dict[f"spars_{s}"] = train_res[str(s)]['sparsity']
        logger_dict["best_acc"] = best_acc
        logger_dict["pr_decay"] = pr_decay.get_dr()
        if args.pruner == "momentum-iter":
            logger_dict["regrow-m"] = mask.curr_gm
        
        print_table(logger_dict.values(), logger_dict.keys(), epoch, logger)
        
        if args.wandb:
            wandb_logger.log(logger_dict)
        
        print(need_time)


if __name__ == '__main__':
    main()
