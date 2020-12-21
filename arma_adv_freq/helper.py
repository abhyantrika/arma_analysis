import torch 
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torchvision
from torchvision import transforms

from PIL import Image
import numpy as np

import freq_helper
import cv2
import dct
import pdb
import os
import glob,copy
import pickle,math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 


def save_tensor(a,filename,norm=True):

	inv_normalize = transforms.Normalize(
		mean=[-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010],
		std=[1/0.2023, 1/0.1994, 1/0.2010]
	)
	
	if norm:
		a = inv_normalize(a)

	a = a.cpu().detach().numpy()
	a = np.squeeze(a)
	a = np.transpose(a,(1,2,0))

	#pdb.set_trace()
	cv2.imwrite(filename,a*255.0)


def get_freq_image(og_img,n_freq,block_freq=None,device=None):
	dct_img = dct.batch_dct(og_img,n_freq=n_freq,block_freq=block_freq,device=device)
	idct_img = dct.batch_idct(dct_img,device=device)	
	
	#Threshold.
	#idct_img[idct_img<0] = 0
	#idct_img[idct_img>1] = 1

	return idct_img

# def write_logs(log_folder,)
def plt_dynamic(fig,x, y, ax, colors=['b']):
	for color in colors:
		if y is None:
			ax.plot(x,c=color)
		else:	
			ax.plot(x, y, color)
	fig.canvas.draw()

def save_params(args):
	with open(args.save_dir+'/args.pkl', 'wb') as fp:
		pickle.dump(args,fp)

	with open(args.save_dir+'/args.txt','w') as fout:
		fout.write(str(args)+'\n')

def check_create_dir(folder):
	if not os.path.exists(folder):
		os.mkdir(folder)

#def load_checkpoint(path):
#	ckpt = torch.load(path)


def adjust_learning_rate(optimizer, epoch, args):
	"""Decay the learning rate based on schedule"""
	lr = args.lr

	if args.cos is None and args.schedule is None:
		return optimizer

	print('LR schedule')
	if args.cos is not None :  # cosine lr schedule
		lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))

	elif args.schedule is not None:  # stepwise lr schedule
		#schedule = args.schedule 
		# print(args.schedule)
		args.schedule = list(map(int,args.schedule))

		for milestone in args.schedule:
			lr *= 0.1 if epoch >= milestone else 1.

	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

	return optimizer


# def get_loaders(args):

# 	import dct_dataloader

# 	transform_train = transforms.Compose([
# 		transforms.RandomCrop(32, padding=4),
# 		transforms.RandomHorizontalFlip(),
# 		transforms.ToTensor(),
# 		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# 	])

# 	transform_test = transforms.Compose([
# 		transforms.ToTensor(),
# 		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# 	])


# 	f_cifar_train_dataset = dct_dataloader.normal_cifar(transform_train,test=False)
# 	f_cifar_test_dataset = dct_dataloader.normal_cifar(transform_test,test=True)

# 	trainloader = torch.utils.data.DataLoader(f_cifar_train_dataset, batch_size=args.batch_size,\
# 	 shuffle=True,num_workers=args.num_workers)
# 	testloader = torch.utils.data.DataLoader(f_cifar_test_dataset, batch_size=args.batch_size, \
# 		shuffle=False,num_workers=args.num_workers)

# 	return trainloader,testloader

def get_loaders(args):

	transform_train = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	transform_test = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	cifar_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,transform=transform_train)
	cifar_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

	trainloader = torch.utils.data.DataLoader(cifar_trainset, batch_size=args.batch_size,\
	 shuffle=True,num_workers=args.num_workers)
	testloader = torch.utils.data.DataLoader(cifar_testset, batch_size=args.batch_size, \
		shuffle=False,num_workers=args.num_workers)

	return trainloader,testloader

def plot_distances(dic,filename,xlim=None,ylim=None,legend=None):

	for k in dic.keys():
		plt.plot(dic[k],label=str(k))

		if xlim is not None:
			plt.xlim(0,xlim)
		if ylim is not None:
			plt.ylim(0,ylim)

	if legend is not None:
		plt.legend(loc=legend)

	plt.savefig(filename)
	plt.clf()

def get_batchwise(dist,args):

	"""	
		convert imagewise values to batch wise.
	"""

	batch_dist = copy.deepcopy(dist)
	batch_size = args.batch_size
	for k in batch_dist.keys():
		dist_list = batch_dist[k]
		temp = []
		i=0
		while i < len(dist_list):
			if (i + batch_size) <= len(dist_list):
				temp.append(sum(dist_list[i:batch_size+i]))
			i+=batch_size
		batch_dist[k] = temp

	return batch_dist

def save_dict(dist,filename):
	with open(filename,'wb') as fout:
		pickle.dump(dist,filename)



def save_batch(images,folder='adv_analysis/',adv=True,norm=True):
	if not os.path.exists(folder):
		os.mkdir(folder)

	inv_normalize = transforms.Normalize(
		mean=[-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010],
		std=[1/0.2023, 1/0.1994, 1/0.2010]
	)

	for i in range(len(images)):
		img = torch.squeeze(images[i])
		if norm:
			img = inv_normalize(img)
		img = img.cpu().detach().numpy()
		img = img.transpose(1,2,0)
		img = img * 255.0

		if adv:
			cv2.imwrite(folder+'/adv_img_'+str(i)+'.png',img)
		else:
			cv2.imwrite(folder+'/img_'+str(i)+'.png',img)



def get_freq_bands(og_img,device='cuda'):

	img_freq_1 = get_freq_image(og_img,n_freq=(0,15),device=device)
	img_freq_2 = get_freq_image(og_img,n_freq=(16,32),device=device)
	img_freq_3 = get_freq_image(og_img,n_freq=(33,47),device=device)
	img_freq_4 = get_freq_image(og_img,n_freq=(48,63),device=device)
	images_freq = [img_freq_1,img_freq_2,img_freq_3,img_freq_4]

	images_freq = torch.stack(images_freq,dim=1)

	return images_freq



def get_all_freq(og_img,device='cuda'):

	images_freq = []
	for i in range(64):
		im = get_freq_image(og_img,n_freq=(i,i),device=device)
		im = im.detach().cpu()
		images_freq.append(im)

	images_freq = torch.stack(images_freq)
	images_freq = torch.transpose(images_freq,1,0)

	#pdb.set_trace()

	return images_freq		

def block_freq(freq_images,freq):

	#for i in range(len(freq_images)):
	#	freq_images[i]

	freq_images[:,freq,:,:] = 0

	return freq_images
	

def to_numpy(img):
	"""
		Takes in tensor in 1,C,H,W format and returns H,W,C format in numpy
	"""

	img = torch.squeeze(img)
	img = img.cpu().detach().numpy()
	img = np.transpose(img,(1,2,0))

	return img

def to_tensor(img,device='cuda'):
	"""
		Takes in numpy image of H,W,C and converts to tensor of C,H,W
	"""
	img = torch.from_numpy(img)
	H,W,C = img.shape

	#img = img.permute([2,1,0])
	img = img.permute([2,0,1])

	return img


def save_tensor(tensor,filename='temp.png'):
	"""
		Tensor is in C,H,W. Convert to H,W,C numpy and save.
	"""
	tensor = torch.squeeze(tensor)
	img = to_numpy(tensor)
	cv2.imwrite(filename,img*255.0)



