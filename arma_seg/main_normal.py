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

from models import resnet,resnet_dilated,resnet18_fcn

import augmentations as aug 

import arma_network 

def main(args):

    if args.debug:
        import pdb;
        pdb.set_trace();

    tb_dir = args.exp_name+'/tb_logs/'
    ckpt_dir = args.exp_name + '/checkpoints/'

    if not os.path.exists(args.exp_name):
        os.mkdir(args.exp_name)
        os.mkdir(tb_dir)
        os.mkdir(ckpt_dir)

    #writer = SummaryWriter(tb_dir+'{}'.format(args.exp_name), flush_secs=10)
    writer = SummaryWriter(tb_dir, flush_secs=10)

    # create model
    print("=> creating model: ")
    os.system('nvidia-smi')
    #model = models.__dict__[args.arch]()

    #model = resnet_dilated.Resnet18_32s(num_classes=21)
    print(args.no_pre_train,' pretrain')
    #model = resnet18_fcn.Resnet18_fcn(num_classes=args.n_classes,pre_train=args.no_pre_train)

    model_map = {
        'deeplabv3_resnet18': arma_network.deeplabv3_resnet18,
        'deeplabv3_resnet50': arma_network.deeplabv3_resnet50,
        'fcn_resnet18': arma_network.fcn_resnet18,
        #'deeplabv3_resnet101': network.deeplabv3_resnet101,
        # 'deeplabv3plus_resnet18': network.deeplabv3plus_resnet18,
        # 'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        # 'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101
    }
    
    model = model_map['deeplabv3_resnet50'](arma=False,num_classes=args.n_classes)

    model = model.cuda()
    model = nn.DataParallel(model)


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
    augmentations = aug.Compose([aug.RandomCrop(512),aug.RandomHorizontallyFlip(5),\
        aug.RandomRotate(30),aug.RandomSizedCrop(512)])

    my_dataset = pascalVOCLoader(args=args,root=args.data,sbd_path=args.data,\
        augmentations=augmentations)

    my_dataset.get_loaders()

    init_weight_filename ='initial_state.pth.tar'
    helper.save_checkpoint(args,model,optimizer,custom_name=init_weight_filename)

    with open(args.exp_name+'/'+'args.pkl','wb') as fout:
        pickle.dump(args,fout)


    best_iou = -100.0
    for epoch in range(args.start_epoch, args.epochs):

        helper.adjust_learning_rate(optimizer, epoch, args)

        train_loss = trainer.train(my_dataset.train_loader,model,optimizer,epoch,args,writer)
        val_loss,scores,class_iou,running_metrics_val = trainer.validate(my_dataset.val_loader, model,epoch,args,writer)
        
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

    with open(args.exp_name+'/running_metric.pkl','wb') as fout:
        pickle.dump(running_metrics_val,fout)


if __name__ == '__main__':
    args = args.get_args()
    main(args)
