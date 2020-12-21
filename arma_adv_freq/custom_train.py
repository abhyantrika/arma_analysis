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

import numpy as np

#sys.path.append('/vulcanscratch/shishira/lottery_stuff/LTH/')

from dataloader import Cifar10,Cifar100
#import custom_helper
#import custom_trainer

import args 
import trainer 
import helper

from utils.VGG import *
from utils.ResNet import *


import pdb
import lth
import pdb 

def train_model(args,model,dataset,writer=None,n_rounds=1,lth_pruner=None):


	root = args.exp_name + '/checkpoints/' 

	criterion = nn.CrossEntropyLoss().cuda()
	optimizer = torch.optim.SGD(model.parameters(), args.lr,momentum=args.momentum,
								weight_decay=args.weight_decay)

	#Save initial weight files. 
	init_weight_filename = args.exp_name + '/checkpoints/' + 'initial_state.pth.tar'
	helper.save_checkpoint(args,model,optimizer,init_weight_filename)	
	
	for cur_round in range(n_rounds):

		best_acc = 0
		for epoch in range(args.start_epoch, args.epochs): 

			helper.adjust_learning_rate(optimizer, epoch, args)

			train_top1, train_top5, train_loss,model = trainer.train(dataset.train_loader,model,criterion,\
				optimizer,epoch,args,lth_pruner,cur_round,mask_applied=args.mask_applied)
			
			val_top1,val_top5,val_loss = trainer.validate(dataset.test_loader, model, criterion, args)

			if writer is not None:
				writer.add_scalar("loss/train/"+str(cur_round), train_loss, epoch)
				writer.add_scalar("top1/train/"+str(cur_round), train_top1, epoch)
				writer.add_scalar("top5/train/"+str(cur_round), train_top5, epoch)

				writer.add_scalar("loss/val/"+str(cur_round), val_loss, epoch)
				writer.add_scalar("top1/val/"+str(cur_round), val_top1, epoch)
				writer.add_scalar("top5/val/"+str(cur_round), val_top5, epoch)            

			if val_top1 >= best_acc:
				best_acc = val_top1
				is_best = True
				filename = root + str(cur_round) + '_model_best.pth'
				helper.save_checkpoint(args,model,optimizer,filename)

			filename = root + str(cur_round) + '_current.pth'
			helper.save_checkpoint(args,model,optimizer,filename,epoch=epoch)
			filename = root + str(cur_round) + '_mask.pkl'

			if epoch in [0,1,2,3]:
				#Save early epochs for late resetting.
				filename = root + 'epoch_'+str(epoch)+ '_model.pth'
				helper.save_checkpoint(args,model,optimizer,filename,epoch=epoch)


def main(args):

	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	if args.debug:
		import pdb;
		pdb.set_trace();

	tb_dir = args.exp_name + '/tb_logs/'
	ckpt_dir = args.exp_name + '/checkpoints/'

	helper.check_create_dir(args.exp_name)
	helper.check_create_dir(tb_dir)
	helper.check_create_dir(ckpt_dir)

	print("writing to : ",tb_dir+'{}'.format(args.exp_name))

	writer = SummaryWriter(tb_dir, flush_secs=10)


	# if args.dataset == 'cifar100':
	# 	my_dataset = Cifar100(args)
	# 	model = resnet.resnet18(num_classes=100)		
	# else:
	# 	my_dataset = Cifar10(args)
	# 	#model = resnet.resnet18(num_classes=10)		
	# 	print('Default model set to VGG')

	#print('loading vgg default with ARMA')

	#model = VGG('VGG11',True,args.dataset,0,3,3)

	if 'vgg' in  args.model_arch.lower():
		model = VGG(args.model_arch,True,args.dataset,0,3,3)
	elif 'res' in args.model_arch.lower():	
		model = ResNet_(args.model_arch,True, args.dataset,0,3,3)

	print("loading ",args.model_arch)	

	if args.dataset == 'CIFAR10':
		my_dataset = Cifar10(args)
	elif args.dataset == 'CIFAR100':
		my_dataset = Cifar100(args)

	model = model.to(device)

	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)

	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			model,optimizer = helper.load_checkpoint(args,model,optimizer)
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	#This takes time, but optimizes to crazy speeds once input is fixed. 
	cudnn.benchmark = True
	
	my_dataset.get_loaders()


	for param in model.parameters():
		param.requires_grad = True

	train_model(args,model,my_dataset,writer=writer,n_rounds=1)


if __name__ == '__main__':
	args = args.get_args()
	main(args)
