import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

#from main import *


class Cifar10(object):
	def __init__(self,args,transform=None,boosting=False,debug=False):
		self.transform = transform
		self.args = args
		self.debug = debug 

		if self.transform is None:
			print("Using standard transform")
			
			#self.transform = transforms.Compose([transforms.ToTensor(),\
			#	transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))])


			self.transform_train = transforms.Compose([
			    transforms.RandomCrop(32, padding=4),
			    transforms.RandomHorizontalFlip(),
			    transforms.ToTensor(),
			    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])

			self.transform_test = transforms.Compose([
			    transforms.ToTensor(),
			    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])


	def get_loaders(self,train_shuffle=True):
		self.train_dataset = datasets.CIFAR10('../data', train=True, download=True,transform=self.transform_train)
		self.test_dataset = datasets.CIFAR10('../data', train=False, transform=self.transform_test)     
		

		self.train_loader = torch.utils.data.DataLoader(
			self.train_dataset, batch_size=self.args.batch_size, shuffle= train_shuffle,
			num_workers=self.args.workers, pin_memory=True, drop_last=True)

		self.test_loader = torch.utils.data.DataLoader(self.test_dataset,batch_size=self.args.batch_size,\
			shuffle=False,num_workers=self.args.workers, pin_memory=True)
		
		return self.train_loader,self.test_loader


class Cifar100(object):
	def __init__(self,args,transform=None,boosting=False):
		self.transform = transform
		self.args = args

		if self.transform is None:
			print("Using standard transform")

			self.transform_train = transforms.Compose([
			    transforms.RandomCrop(32, padding=4),
			    transforms.RandomHorizontalFlip(),
			    transforms.ToTensor(),
			    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])

			self.transform_test = transforms.Compose([
			    transforms.ToTensor(),
			    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])


	def get_loaders(self,train_shuffle=True):
		self.train_dataset = datasets.CIFAR100('../data', train=True, download=True,transform=self.transform_train)
		self.test_dataset = datasets.CIFAR100('../data', train=False, transform=self.transform_test)     
		
		self.train_loader = torch.utils.data.DataLoader(
			self.train_dataset, batch_size=self.args.batch_size, shuffle= train_shuffle,
			num_workers=self.args.workers, pin_memory=True, drop_last=True)

		self.test_loader = torch.utils.data.DataLoader(self.test_dataset,batch_size=self.args.batch_size,\
			shuffle=False,num_workers=self.args.workers, pin_memory=True)
		
		return self.train_loader,self.test_loader


if __name__ == '__main__':
	laoder = Cifar100()
