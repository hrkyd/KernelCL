#!/usr/bin/env python

# This code is adapted from the original implementation of "simsiam/main_lincls.py" on GitHub (https://github.com/facebookresearch/simsiam)
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

import arch.resnet as resnet
import optim.lr_scheduler as lr_scheduler
import utils.builder as builder


parser = argparse.ArgumentParser()
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', default='imagenet', type=str)
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4096, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

# additional configs:
parser.add_argument('--pretrained', default='', type=str,
                    help='path to simsiam pretrained checkpoint')
parser.add_argument('--lars', action='store_true',
                    help='Use LARS')
parser.add_argument('--image_size', default=224, type=int)
parser.add_argument('--log_path', default='./', type=str)
parser.add_argument('--conv1_type', default='imagenet', type=str)
parser.add_argument('--no_maxpool', action='store_true')
parser.add_argument('--type', default=None, type=str)
parser.add_argument('--subset', default=None, nargs='*', type=int)

best_acc1 = 0


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


    if torch.cuda.is_available():
        args.gpu = 'cuda:{}'.format(torch.cuda.current_device())
    main_worker(args.gpu, args)

class LinearHead(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(LinearHead, self).__init__()
        self.head = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, x):
        x = self.head(x)
        return x

def load_model(args, checkpoint):
    # create model
    num_classes = {
        'cifar10': 10,
        'stl10': 10,
        'imagenet100': 100,
    }
    print("=> creating model '{}'".format(args.arch))
    if "resnet" in args.arch:
        model = resnet.__dict__[args.arch](num_classes=num_classes[args.dataset],  conv1_type=args.conv1_type, no_maxpool=args.no_maxpool)
    else:
        raise NotImplementedError

    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    return model

def main_worker(gpu, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            model = load_model(args, checkpoint)

            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                if k.startswith('encoder') and not k.startswith('encoder.fc'):
                    state_dict[k[len("encoder."):]] = state_dict[k]
                del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {'fc.weight', 'fc.bias'}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))
            raise AttributeError

    init_lr = args.lr * args.batch_size / 256

    if args.gpu is not None:
        print('Use single GPU')
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        raise NotImplementedError

    print(model)

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2

    optimizer = torch.optim.SGD(parameters, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
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
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    val_transform = transforms.Compose([
            transforms.Resize(int(args.image_size * 8 / 7)),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize,
        ])

    print('dataset: {}'.format(args.dataset))
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(args.data, train=True, transform=train_transform)
    elif args.dataset == 'stl10':
        train_dataset = datasets.STL10(args.data, split='train', transform=train_transform)
    elif args.dataset == 'imagenet100':
        train_dataset = datasets.ImageFolder(traindir, transform=train_transform)
    else:
        raise NotImplementedError

    if args.dataset == 'cifar10':
        val_dataset = datasets.CIFAR10(args.data, train=False, transform=val_transform)
    elif args.dataset == 'stl10':
        val_dataset = datasets.STL10(args.data, split='test', transform=val_transform)
    elif args.dataset == 'imagenet100':
        val_dataset = datasets.ImageFolder(valdir, transform=val_transform)
    else:
        raise NotImplementedError

    train_sampler = None
    val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=256, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    warmup_epochs = 0
    base_lr = init_lr
    final_lr = 0.0
    num_iter = len(train_loader.dataset) // args.batch_size
    print("number of iterations per one epoch: {}".format(num_iter))
    lr = lr_scheduler.LR_Scheduler(warmup_epochs, args.epochs, init_lr, base_lr, final_lr, num_iter)
    assert len(lr.warmup_lr_schedule) == 0

    print(args)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, args, lr)

        if (epoch + 1) % 5 == 0:
            acc1, acc5 = validate(val_loader, model, criterion, args)

            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            save_checkpoint({
                'args': args,
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'acc1': acc1,
                'acc5': acc5,
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args)
            if epoch == args.start_epoch:
                sanity_check(model.state_dict(), args.pretrained, args)

    acc_save_path = './lincls_acc/{}/{}_log.csv'.format(args.type, args.dataset)

    if not os.path.exists('./lincls_acc'):
        os.makedirs('./lincls_acc')

    if not os.path.exists('./lincls_acc/{}'.format(args.type)):
        os.makedirs('./lincls_acc/{}'.format(args.type))

    if not os.path.exists(acc_save_path):
        with open(acc_save_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerows([['log_path', 'acc1', 'acc5', 'best_acc1']])

    acc_log = [[args.log_path, acc1, acc5, best_acc1]]
    with open(acc_save_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerows(acc_log)


def train(train_loader, model, criterion, optimizer, epoch, args, lr):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    """
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        lr.step(optimizer)
        if epoch == 0 and i == 0:
            assert lr.cur_lr() == lr.init_lr

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            output = model(images)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, args, filename='./checkpoint.pth.tar'):
    filename = '{}/checkpoint.pth.tar'.format(args.log_path)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}/model_best.pth.tar'.format(args.log_path))


def sanity_check(state_dict, pretrained_weights, args):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        k_pre = 'encoder.' + k[len(''):] \
            if k.startswith('') else 'encoder.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


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


if __name__ == '__main__':
    main()
