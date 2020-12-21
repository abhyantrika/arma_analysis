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


def main(args):


	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	if 'vgg' in  args.model_arch.lower():
		model = VGG(args.model_arch,True,args.dataset,0,3,3)
	elif 'res' in args.model_arch.lower():	
		model = ResNet_(args.model_arch,True, args.dataset,0,3,3)


	my_dataset = Cifar10(args)



	model = model.to(device)

	#if torch.cuda.device_count() > 1:
	model = nn.DataParallel(model)

	my_dataset.get_loaders()

	model_path = args.resume 
	model,_,_ = helper.load_checkpoint(args,model,optimizer=None,path=None)

	criterion = nn.CrossEntropyLoss()
	top_1_acc,_,_ = trainer.validate(my_dataset.test_loader, model, criterion, args)


if __name__ == '__main__':
	args = args.get_args()
	main(args)