#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil,pickle
import time,sys
import warnings

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
#import torchvision.models as models
import torchvision


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

import copy
import helper
import numpy as np

from collections import defaultdict

import pdb
import cv2
import argparse

import dct

from metric import Metric_counter
import utils.config as cf

from utils.AlexNet import *
from utils.VGG import *
from utils.ResNet import *

import baseline_vgg,baseline_resnet

sys.path.append('/vulcanscratch/shishira/adv_stuff/hi_freq/cifar/')

import dct_stats


parser = argparse.ArgumentParser()

parser.add_argument('--model_path',help='path to trained  model',default='')
parser.add_argument('--exp_dir',help='path to experiment folder',default='')

parser.add_argument('--eps',help='model eps',default=0.031,type=float)
parser.add_argument('--kernel_size',help='kernel size for gaussian',default=None,type=int)
parser.add_argument('--sigma',help='Sigma for gaussian',default=0,type=float)

parser.add_argument('--attack',help='attack_name',default='fgsm')
parser.add_argument('--iters',help='model steps',default=20,type=int)

parser.add_argument("--batch_size", type=int,default=1)
parser.add_argument("--num_workers", type=int,default=16)

parser.add_argument("--save_folder", type=str,default='attack_plots/temp')

#parser.add_argument("--no-arma", dest="arma", action="store_false",help="Do not use arma layer in the model.")

parser.add_argument('--model_arch',type=str,help='architecture',default='VGG11')

parser.add_argument("--baseline", dest="baseline", action="store_true",help="use baseline models from lth baselines")
parser.add_argument('--dataset',type=str,help='CIFAR10 or CIFAR100',default='CIFAR10')	

args = parser.parse_args()



def get_y_channel(og_img,stats_path='stats/cstats.pt',device='cuda'):
	#print("Loading stats from: ",stats_path)
	stat_class = dct_stats.DCTStats(stats_path)

	ycb_images = dct.images_to_batch(og_img,stat_class, device=device)
	y_images = ycb_images[:,0,:,:]

	#Check if y_channel images can be recovered.
	return y_images,ycb_images

def ycb_sanity(img,filename,stats_path='stats/cstats.pt',device='cuda',norm=True):

	stat_class = dct_stats.DCTStats(stats_path)
	rgb_image = dct.batch_to_images(img,stat_class,device=device)


	inv_normalize = transforms.Normalize(
		mean=[-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010],
		std=[1/0.2023, 1/0.1994, 1/0.2010]
	)

	img = torch.squeeze(rgb_image)

	if norm:
		img = inv_normalize(img)

	img = img.cpu().detach().numpy()
	img = img.transpose(1,2,0)
	img = img * 255.0	

	cv2.imwrite(filename,img)


def pgd_attack(model, images, labels, eps,iters):

	device = 'cuda'

	for param in model.parameters():
		param.requires_grad = True

	model.requires_grad_(True)	

	n,c,h,w = images.shape

	mean=torch.tensor([0.4914, 0.4822, 0.4465]).cuda()
	std= torch.tensor([0.2023, 0.1994, 0.2010]).cuda()
	lower,upper = -mean/std, (1-mean)/std
	lower = lower.reshape(1,c,1,1)
	upper = upper.reshape(1,c,1,1)

	criterion = nn.CrossEntropyLoss()
	images.requires_grad = True

	for i in range(iters):
		images.requires_grad = True

		outputs = model(images)

		img_min = images.min().item()
		img_max = images.max().item()

		model.zero_grad()
		cost = criterion(outputs, labels).to(device)
		cost.backward()


		delta = eps*images.grad.sign()
		delta = torch.clamp(delta, min=-eps, max=eps)

		images = images + delta 

		images = torch.where(images < lower,lower,torch.where(images > upper,upper,images))	

		images = images.detach()
		
		#attack_images = images + delta 

		#Each channel clampled separately
		#attack_images = torch.where(attack_images < lower,lower,torch.where(attack_images> upper,upper,attack_images))

	return images,images.grad


def fgsm_attack(model,images,labels,eps):

	device = 'cuda'

	for param in model.parameters():
		param.requires_grad = True

	model.requires_grad_(True)

	#images.requires_grad = True

	n,c,h,w = images.shape

	mean=torch.tensor([0.4914, 0.4822, 0.4465]).cuda()
	std= torch.tensor([0.2023, 0.1994, 0.2010]).cuda()
	lower,upper = -mean/std, (1-mean)/std

	#Need this to bound the epsilon

	lower = lower.reshape(1,c,1,1)
	upper = upper.reshape(1,c,1,1)


	criterion = nn.CrossEntropyLoss()
	images.requires_grad = True

	outputs = model(images)

	img_min = images.min().item()
	img_max = images.max().item()

	#print(img_min,img_max,lower,upper)

	model.zero_grad()
	cost = criterion(outputs, labels).cuda()
	cost.backward()


	delta = eps*images.grad.sign()
	delta = torch.clamp(delta, min=-eps, max=eps)

	attack_images = images + delta 

	#Each channel clampled separately
	attack_images = torch.where(attack_images < lower,lower,torch.where(attack_images> upper,upper,attack_images))


	return attack_images,images.grad


def apply_gaussian_blur(adv_img,kernel_size=5,sigma=0):

	#if sigma=0, cv2.GaussianBlur caluclates sigma from kernel size.

	N,ch,H,W = adv_img.shape

	for i in range(N):
		img = helper.to_numpy(adv_img[i])
		img = cv2.GaussianBlur(img,(kernel_size,kernel_size),sigmaX=sigma) 
		adv_img[i] = helper.to_tensor(img)

	return adv_img


# if args.baseline:
# 	import baseline_vgg
# 	model = baseline_vgg.vgg11()
# else:

baseline_dict = {'VGG11':baseline_vgg.vgg11(),\
'VGG19':baseline_vgg.vgg19(),'ResNet18':baseline_resnet.ResNet18(),'ResNet50':baseline_resnet.ResNet50()}

if 'vgg' in  args.model_arch.lower():
	if args.baseline:
		model = baseline_dict[args.model_arch]
	else:
		model = VGG(args.model_arch,True,args.dataset,0,3,3)

elif 'res' in args.model_arch.lower():	
	if args.baseline:
		model = baseline_dict[args.model_arch]
	else:
		model = ResNet_(args.model_arch,True, args.dataset,0,3,3)


# model = resnet.resnet18()
model = nn.DataParallel(model)
ckpt = torch.load(args.model_path)

try:
	state_dict = ckpt['model_state_dict']
except:
	state_dict = ckpt['state_dict']
	
model.load_state_dict(state_dict)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

trainloader,testloader = helper.get_loaders(args)


inv_normalize = transforms.Normalize(
	mean=[-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010],
	std=[1/0.2023, 1/0.1994, 1/0.2010]
)

normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

# Now remove normalization and repeat.
metric = Metric_counter(args)
correct = 0
total = 0
test_loss  = 0

for batch_idx, (img,targets) in enumerate(testloader): 
	img,targets = img.cuda(),targets.cuda()

	#B;urring needs to happen after adv example is generated!

	if args.attack == 'fgsm':
		attacked_img,grad = fgsm_attack(model,img,targets,args.eps)
	elif args.attack == 'pgd':
		attacked_img,grad = pgd_attack(model,img,targets,args.eps,args.iters)

	if args.kernel_size is not None:
		attacked_img = apply_gaussian_blur(attacked_img,kernel_size=args.kernel_size,sigma=args.sigma)

	#Our analysis.
	og_img = img.clone()
	adv_img = attacked_img.clone()

	for k in range(len(img)):
		og_img[k] = inv_normalize(img[k])
		adv_img[k] = inv_normalize(attacked_img[k])

	criterion = nn.CrossEntropyLoss()

	adv_img_block_freq = helper.get_freq_image(adv_img,(0,63),block_freq=[57,63],device='cuda')
	#blocked freq image is in 0-1.
	#Normalize and then pass.
	for i in range(len(adv_img_block_freq)):
		adv_img_block_freq[i] = normalize(adv_img_block_freq[i])

	#reconstruct attacked_img
	outputs = model(adv_img_block_freq)
	loss = criterion(outputs, targets).cuda()

	adv_img = adv_img_block_freq.clone()
	for k in range(len(img)):
		adv_img[k] = inv_normalize(attacked_img[k])

	og_img_all_freq = helper.get_all_freq(og_img.clone())
	og_img_bands = helper.get_freq_bands(og_img.clone())
	og_img_y,og_ycb = get_y_channel(og_img.clone())
	og_img_y.unsqueeze_(1)
	og_img_y_freq_bands = helper.get_freq_bands(og_img_y.clone())
	og_img_y_all_freq = helper.get_all_freq(og_img_y.clone())


	attacked_img_all_freq = helper.get_all_freq(adv_img.clone())
	attacked_img_bands = helper.get_freq_bands(adv_img.clone())
	attacked_img_y,attacked_ycb = get_y_channel(adv_img.clone())
	attacked_img_y.unsqueeze_(1)
	attacked_img_y_freq_bands = helper.get_freq_bands(attacked_img_y.clone())
	attacked_img_y_all_freq = helper.get_all_freq(attacked_img_y.clone())


	test_loss += loss.item()
	_, predicted = outputs.max(1)
	total += targets.size(0)
	correct += predicted.eq(targets).sum().item()

	info = 'batch_idx ' + str(batch_idx) + '  Loss: %.3f | Acc: %.3f%% (%d/%d)'\
				 % (test_loss/(batch_idx+1), 100.*correct/total, correct, total)

	if batch_idx%2==0:
		print(info)

	metric.add(og_img_all_freq,og_img_y_all_freq,attacked_img_all_freq,attacked_img_y_all_freq)
	metric.add_bands(og_img_bands,og_img_y_freq_bands,attacked_img_bands,attacked_img_y_freq_bands)


if args.kernel_size is not None:
	save_folder = args.save_folder+'_'+str(args.eps) + '_'+str(args.kernel_size) + '_' + str(args.sigma)
else:
	save_folder = args.save_folder+'_'+str(args.eps)

if not os.path.exists(save_folder+'/'):
	os.mkdir(save_folder+'/')

metric.create_batchwise()
metric.plot(save_folder)

acc = correct / total
print("acc: ",correct/total)

with open(save_folder+'/'+'metric.pkl','wb') as fout:
	pickle.dump(metric,fout)

with open(save_folder+'/'+'log.txt','w') as fout:
	fout.write(str(acc))