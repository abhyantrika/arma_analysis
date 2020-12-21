#from main import *

import torch.nn as nn 
import torch

import sys
import helper
import time 
import pdb 


def train(train_loader, model, criterion, optimizer, epoch, args,lth_pruner,cur_round,mask_applied):
	batch_time = helper.AverageMeter('Time', ':6.3f')
	data_time = helper.AverageMeter('Data', ':6.3f')
	losses = helper.AverageMeter('Loss', ':.4e')
	top1 = helper.AverageMeter('Acc@1', ':6.2f')
	top5 = helper.AverageMeter('Acc@5', ':6.2f')
	progress = helper.ProgressMeter(
		len(train_loader),
		[batch_time, data_time, losses, top1, top5],
		prefix="Epoch: [{}]".format(epoch))

	# switch to train mode
	model.train()

	end = time.time()
	for i, (images, target) in enumerate(train_loader):
		
		data_time.update(time.time() - end)
		images,target = images.cuda(),target.cuda()
		output = model(images)

		# import pdb;
		# pdb.set_trace()

		loss = criterion(output, target)

		acc1, acc5 = helper.accuracy(output, target, topk=(1, 5))
		losses.update(loss.item(), images[0].size(0))
		top1.update(acc1[0], images[0].size(0))
		top5.update(acc5[0], images[0].size(0))

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()

		#aa = list(model.parameters())
		#print(aa[0].mean())

		#If in prune mode, block out gradients.
		#Basically after LTH pruning starts,maintain zero weights at zero.
		if cur_round >0 or mask_applied:
			for k, (name,param) in enumerate(model.named_parameters()):
				#if 'weight' in name:
				if name in lth_pruner.mask:
					weight_copy = param.data.abs().clone()
					mask = weight_copy.gt(0).float().cuda()
					param.grad.data.mul_(mask)

		optimizer.step()
		batch_time.update(time.time() - end)
		end = time.time()

		#pdb.set_trace()
		if i % 10 == 0:
			progress.display(i)

	return top1.avg, top5.avg,losses.avg,model    



def validate(val_loader, model, criterion, args):
	batch_time = helper.AverageMeter('Time', ':6.3f')
	losses = helper.AverageMeter('Loss', ':.4e')
	top1 = helper.AverageMeter('Acc@1', ':6.2f')
	top5 = helper.AverageMeter('Acc@5', ':6.2f')
	progress = helper.ProgressMeter(
		len(val_loader),
		[batch_time, losses, top1, top5],
		prefix='Test: ')

	# switch to evaluate mode
	model.eval()

	with torch.no_grad():
		end = time.time()
		#print(len(val_loader),args.batch_size)

		for i, (images, target) in enumerate(val_loader):
			images,target = images.cuda(),target.cuda()
			output = model(images)
			
			loss = criterion(output, target)
			# measure accuracy and record loss
			acc1, acc5 = helper.accuracy(output, target, topk=(1, 5))
			losses.update(loss.item(), images.size(0))
			top1.update(acc1[0], images.size(0))
			top5.update(acc5[0], images.size(0))

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % 10 == 0:
				progress.display(i)

		# TODO: this should also be done with the helper.ProgressMeter
		print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
			  .format(top1=top1, top5=top5))

	return top1.avg, top5.avg,losses.avg  


def batch_boost_train(train_loader, model,weak_model,samplewise_criterion,\
	optimizer_1,optimizer_2,epoch, args):

	batch_time = helper.AverageMeter('Time', ':6.3f')
	data_time = helper.AverageMeter('Data', ':6.3f')
	
	losses_strong = helper.AverageMeter('Loss strong', ':.4e')
	top1_strong = helper.AverageMeter('Acc@1 strong', ':6.2f')
	
	losses_weak = helper.AverageMeter('Loss weak', ':.4e')
	top1_weak = helper.AverageMeter('Acc@1 weak', ':6.2f')


	progress = helper.ProgressMeter(
		len(train_loader),
		[batch_time, data_time, losses_strong,top1_strong, losses_weak,top1_weak],
		prefix="Epoch: [{}]".format(epoch))

	samplewise_criterion_weak = nn.CrossEntropyLoss(reduce='none')

	# switch to train mode
	model.train()
	weak_model.train()

	end = time.time()
	for i, (images, target) in enumerate(train_loader):
		
		data_time.update(time.time() - end)
		images,target = images.cuda(),target.cuda()
		output = model(images)

		samplewise_loss_strong = samplewise_criterion(output, target)

		strong_loss = samplewise_loss_strong.mean()
		acc1, acc5 = helper.accuracy(output, target, topk=(1, 5))
		losses_strong.update(strong_loss.item(), images[0].size(0))
		top1_strong.update(acc1[0], images[0].size(0))
		
		# compute gradient and do SGD step
		optimizer_1.zero_grad()
		strong_loss.backward()

		#Use the per sample loss from strong model as weights for the weak model. 
		#Rescale sample wise loss to [0,1]
		weights = samplewise_loss_strong.clone().detach()
		weights -= weights.min()
		weights /= weights.max()

		output_2 = weak_model(images)
		samplewise_loss_weak = samplewise_criterion_weak(output_2, target)

		#Use it as weightsfor the weak model loss.
		samplewise_loss_weak = samplewise_loss_weak * weights
		weak_loss = samplewise_loss_weak.mean()
		acc1, acc5 = helper.accuracy(output_2, target, topk=(1, 5))
		losses_weak.update(weak_loss.item(), images[0].size(0))
		top1_weak.update(acc1[0], images[0].size(0))
		
		optimizer_2.zero_grad()
		weak_loss.backward()


		#If in prune mode, block out gradients.
		#Basically after LTH pruning starts,maintain zero weights at zero.
		for k, (name,param) in enumerate(model.named_parameters()):
			if 'weight' in name and 'bn' not in name:
				weight_copy = param.data.abs().clone()
				mask = weight_copy.gt(0).float().cuda()
				param.grad.data.mul_(mask)

		for k, (name,param) in enumerate(weak_model.named_parameters()):
			if 'weight' in name and 'bn' not in name:
				weight_copy = param.data.abs().clone()
				mask = weight_copy.gt(0).float().cuda()
				param.grad.data.mul_(mask)


		optimizer_1.step()
		optimizer_2.step()

		#pdb.set_trace()

		batch_time.update(time.time() - end)
		end = time.time()

		if i % 10 == 0:
			progress.display(i)

	return top1_strong.avg, top1_weak.avg,losses_strong.avg,losses_weak.avg,model,weak_model    
