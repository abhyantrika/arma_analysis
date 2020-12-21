#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import sys
import pickle

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
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter


import helper
import trainer
import args
from dataloader import pascalVOCLoader

from models import resnet,resnet_dilated

import augmentations as aug 

def main(args):


    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        print(ngpus_per_node,args.gpu)
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.debug:
        import pdb;
        pdb.set_trace();

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
        
        tb_dir = args.exp_name+'/tb_logs/'
        ckpt_dir = args.exp_name + '/checkpoints/'

        if not os.path.exists(args.exp_name):
            os.mkdir(args.exp_name)
            os.mkdir(tb_dir)
            os.mkdir(ckpt_dir)

        print("writing to : ",tb_dir+'{}'.format(args.exp_name),args.rank,ngpus_per_node)

        #writer = SummaryWriter(tb_dir+'{}'.format(args.exp_name), flush_secs=10)
        writer = SummaryWriter(tb_dir, flush_secs=10)

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model: ")
    #model = models.__dict__[args.arch]()

    model = resnet_dilated.Resnet18_32s(num_classes=21)

    if args.distributed:
        print("distributed")
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")



    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            model,optimizer,args = helper.load_checkpoint(args,model,optimizer)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    #USE this only when batch size is fixed. 
    #This takes time, but optimizes to crazy speeds once input is fixed. 
    cudnn.benchmark = True

    #Load dataloaders
    augmentations = aug.Compose([aug.RandomCrop(512),aug.RandomHorizontallyFlip(5),aug.RandomRotate(30),aug.RandomSizedCrop(512)])
    my_dataset = pascalVOCLoader(args=args,root='/scratch0/shishira/pascal_voc/',sbd_path='/scratch0/shishira/pascal_voc/',\
        augmentations=augmentations)
    my_dataset.get_loaders()

    init_weight_filename ='initial_state.pth.tar'
    helper.save_checkpoint(args,model,optimizer,custom_name=init_weight_filename)

    with open(args.exp_name+'/'+'args.pkl','wb') as fout:
        pickle.dump(args,fout)


    best_iou = -100.0
    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            my_dataset.train_sampler.set_epoch(epoch)

        helper.adjust_learning_rate(optimizer, epoch, args)

        train_loss = trainer.train(my_dataset.train_loader,model,optimizer,epoch,args,writer)
        val_loss,scores,class_iou = trainer.validate(my_dataset.val_loader, model,epoch,args,writer)
        
        if scores["Mean IoU : \t"] >= best_iou:
            best_iou = scores["Mean IoU : \t"]
            is_best = True

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):

            if epoch in [0,1,2,3,4,5,6,7,8]:
                helper.save_checkpoint(args,model,optimizer,epoch,custom_name=str(epoch)+'.pth')

            if args.save_freq is None:
                helper.save_checkpoint(args,model,optimizer,epoch,is_best=is_best,periodic=False)
            else:
                helper.save_checkpoint(args,model,optimizer,epoch,is_best=is_best,periodic=True)


if __name__ == '__main__':
    args = args.get_args()
    main(args)
