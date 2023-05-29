#!/usr/bin/env python

# This code is adapted from the original implementation of "simsiam/main_simsiam.py" on GitHub (https://github.com/facebookresearch/simsiam)
#################################################################
# The information about the licence of the original "simsiam":
# Copyright (c) Facebook, Inc. and its affiliates
# Released under the CC-BY-NC 4.0 license 
# https://github.com/facebookresearch/simsiam/blob/main/LICENSE
#################################################################

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import pickle
import csv

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

import utils.loader as loader
import utils.builder as builder
from kcl.loss import Loss
import arch.resnet as resnet
import optim.lr_scheduler as lr_scheduler


parser = argparse.ArgumentParser()
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512)')
parser.add_argument('--init_lr', default=0.0005, type=float)
parser.add_argument('--base_lr', default=0.05, type=float)
parser.add_argument('--final_lr', default=0.0, type=float)
parser.add_argument('--warmup_epochs', default=10, type=int)
parser.add_argument('--lr_scale', default='linear', type=str)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=0.0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--image_size', default=224, type=int)
parser.add_argument('--dim', default=2048, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--log_path', default='./', type=str)
parser.add_argument('--optimizer', default='sgd', type=str)
parser.add_argument('--lars', action='store_true')
parser.add_argument('--fix-pred-lr', action='store_true',
                    help='Fix learning rate for the predictor')
parser.add_argument('--beta_1', default=0.9, type=float)
parser.add_argument('--beta_2', default=0.999, type=float)

# configs for architectures
parser.add_argument('--no_maxpool', action='store_true')
parser.add_argument('--conv1_type', default=None, type=str)
parser.add_argument('--proj_layer', default=None, type=int)

# KCL specific configs:
parser.add_argument('--band_width', default=None, type=float)
parser.add_argument('--weight', default=None, type=float)
parser.add_argument('--type', default=None, type=str)
parser.add_argument('--no_normalize', action='store_true')

# config
parser.add_argument('--sh_file', default=None, type=str)


def main():
    args = parser.parse_args()

    if args.seed is not None:
        print('fixed seed: {}'.format(args.seed))
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        np.random.seed(args.seed)
    else:
        cudnn.benchmark = True

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    else:
        raise AttributeError

    if args.sh_file is not None:
        config_path = '{}/config.sh'.format(args.log_path)
        shutil.copyfile(args.sh_file + '.sh', config_path)

    if torch.cuda.is_available():
        args.gpu = 'cuda:{}'.format(torch.cuda.current_device())
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    print("=> creating model '{}'".format(args.arch))
    if "resnet" in args.arch:
        model = builder.Model(
            resnet.__dict__[args.arch],
            args.dim, no_maxpool=args.no_maxpool, conv1_type=args.conv1_type, proj_layer=args.proj_layer, method_type=args.type)
    else:
        raise NotImplementedError


    init_lr = args.init_lr
    final_lr = args.final_lr

    if args.lr_scale == 'linear':
        base_lr = args.base_lr * args.batch_size / 256
    else:
        raise NotImplementedError

    if args.gpu is not None:
        print('Use single GPU')
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        raise NotImplementedError
    print(model)

    if args.type in ['gaussian', 'quadratic']:
        criterion = Loss(args).cuda(args.gpu)
    else:
        raise NotImplementedError

    optim_params = model.parameters()

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(optim_params, init_lr, betas=(args.beta_1, args.beta_2), weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    if args.lars:
        print("=> use LARS optimizer.")
        from apex.parallel.LARC import LARC
        optimizer = LARC(optimizer=optimizer, trust_coefficient=.001, clip=False)


    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    if args.image_size > 32:
        augmentation = [
            transforms.RandomResizedCrop(args.image_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        augmentation = [
            transforms.RandomResizedCrop(args.image_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    print('dataset: {}'.format(args.dataset))
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(args.data, train=True, transform=loader.TwoCropsTransform(transforms.Compose(augmentation)))
    elif args.dataset == 'stl10':
        train_dataset = datasets.STL10(args.data, split='train+unlabeled', transform=loader.TwoCropsTransform(transforms.Compose(augmentation)))
    elif args.dataset == 'imagenet100':
        train_dataset = datasets.ImageFolder(traindir, transform=loader.TwoCropsTransform(transforms.Compose(augmentation)))
    else:
        raise NotImplementedError('Dataset not found')


    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    
    num_iter = len(train_loader.dataset) // args.batch_size
    print("number of the images: {}".format(len(train_loader.dataset)))
    print("number of iterations per one epoch: {}".format(num_iter))

    lr = lr_scheduler.LR_Scheduler(args.warmup_epochs, args.epochs, args.init_lr, args.base_lr, args.final_lr, num_iter)

    if not os.path.exists(os.path.join(args.log_path, 'pretrain_log.csv')):
        with open(os.path.join(args.log_path, 'pretrain_log.csv'), 'w') as fp:
            _writer = csv.writer(fp)
            _writer.writerows([['Epoch', ' Loss']])

    print(args)

    for epoch in range(args.start_epoch, args.epochs):

        train(train_loader, model, criterion, optimizer, epoch, args, lr, num_iter)

        if (epoch+1) % 100 == 0:
            save_checkpoint({
                'args': args,
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, False, args, filename='{}/checkpoint_{:04d}.pth.tar'.format(args.log_path, epoch))


def train(train_loader, model, criterion, optimizer, epoch, args, lr, num_iter):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))
    len_train_loader = len(train_loader)

    model.train()

    print('Current Epoch: {} (the number of iterations: {})'.format(epoch, len_train_loader))

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        lr.step(optimizer)
        if epoch == 0 and i == 0:
            assert lr.cur_lr() == lr.init_lr

        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        if args.type in ['gaussian', 'quadratic']:
            z1, z2 = model(x1=images[0], x2=images[1])
            loss, pos, neg = criterion(z1, z2)
        else:
            raise NotImplementedError

        losses.update(loss.item(), images[0].size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    current_lr = lr.cur_lr()
    log = [['Epoch: [{}][{}/{}]'.format(epoch, i, len_train_loader), ' Loss {} (pos:{}, neg:{}), LR {}'.format(loss.item(), pos.item(), neg.item(), current_lr)]]
    
    with open(os.path.join(args.log_path, 'pretrain_log.csv'), 'a') as f:
        writer = csv.writer(f)
        writer.writerows(log)



def save_checkpoint(state, is_best, args, filename='./checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}/model_best.pth.tar'.format(args.log_path))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    main()
