from main import *
import helper

import torch.nn.functional as F

import pdb

def cross_entropy2d(input, target, weight=None, size_average=True):
	n, c, h, w = input.size()
	nt, ht, wt = target.size()

	# Handle inconsistent size between input and target
	if h != ht and w != wt:  # upsample labels
		input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

	input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
	target = target.view(-1)


	loss = F.cross_entropy(
		input, target, weight=weight, size_average=size_average, ignore_index=250
	)
	return loss




def train(train_loader, model, optimizer, epoch, args,writer):
	batch_time = helper.AverageMeter('Time', ':6.3f')
	data_time = helper.AverageMeter('Data', ':6.3f')
	losses = helper.AverageMeter('Loss', ':.4e')
	progress = helper.ProgressMeter(
		len(train_loader),
		[batch_time, data_time, losses],
		prefix="Epoch: [{}]".format(epoch))

	ngpus_per_node = torch.cuda.device_count()
	# switch to train mode
	model.train()

	end = time.time()

	for i, (images, target) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)

		if args.gpu is not None:
			images = images.cuda(args.gpu, non_blocking=True)
		if torch.cuda.is_available():
			target = target.cuda(args.gpu, non_blocking=True)

		output = model(images)
		loss = cross_entropy2d(output, target)

		losses.update(loss.item(), images[0].size(0))
		
		#compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % 10 == 0:
			progress.display(i)

	if not args.multiprocessing_distributed or (args.multiprocessing_distributed
			and args.rank % ngpus_per_node == 0):
		writer.add_scalar("loss/train",loss,epoch)

	return losses.avg


def validate(val_loader, model, epoch,args,writer):
	batch_time = helper.AverageMeter('Time', ':6.3f')
	losses = helper.AverageMeter('Loss', ':.4e')
	progress = helper.ProgressMeter(
		len(val_loader),
		[batch_time, losses],
		prefix='Test: ')

	running_metrics_val = helper.runningScore(args.n_classes)
	# switch to evaluate mode
	model.eval()
	ngpus_per_node = torch.cuda.device_count()


	with torch.no_grad():
		end = time.time()
		#print(len(val_loader),args.batch_size)

		for i, (images, target) in enumerate(val_loader):

			if args.gpu is not None:
				images = images.cuda(args.gpu, non_blocking=True)
			target = target.cuda(args.gpu, non_blocking=True)

			# compute output
			output = model(images)
			loss = cross_entropy2d(output, target)

			# measure accuracy and record loss
			losses.update(loss.item(), images.size(0))

			pred = output.max(1)[1].cpu().numpy()
			gt = target.cpu().numpy()

			running_metrics_val.update(gt, pred)

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % 10 == 0:
				progress.display(i)

	score, class_iou = running_metrics_val.get_scores()
	if not args.multiprocessing_distributed or (args.multiprocessing_distributed
			and args.rank % ngpus_per_node == 0):
		writer.add_scalar("loss/val",losses.avg,epoch)
		for k, v in score.items():
			print(k, v)
			print("{}: {}".format(k, v))
			writer.add_scalar("val_metrics/{}".format(k), v, epoch)

		for k, v in class_iou.items():
			print("{}: {}".format(k, v))
			writer.add_scalar("val_metrics/cls_{}".format(k), v, epoch)

	#running_metrics_val.reset()

	return losses.avg ,score,class_iou,running_metrics_val


def adv_train_free(train_loader, model, criterion,optimizer,global_noise_data,\
	step_size,clip_eps,mean,std,epoch,args):

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

	#Bounds for clamping, if not in 0-1 range. 
	lower,upper = -mean/std, (1-mean)/std
	lower = lower.reshape(1,c,1,1)
	upper = upper.reshape(1,c,1,1)

	end = time.time()
	for i, (images, target) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)

		if args.gpu is not None:
			images = images.cuda(args.gpu, non_blocking=True)
			target = target.cuda(args.gpu,non_blocking=True)

		for j in range(configs.ADV.n_repeats):
			# Ascend on the global noise
			noise_batch = Variable(global_noise_data[0:input.size(0)], requires_grad=True).cuda()
			in1 = input + noise_batch
			
			#in1.clamp_(0, 1.0)
			#in1.sub_(mean).div_(std)
			in1 = torch.where(in1 < lower,lower,torch.where(in1> upper,upper,in1))

			output = model(in1)
			loss = criterion(output, target)
			
			prec1, prec5 = accuracy(output, target, topk=(1, 5))
			losses.update(loss.item(), input.size(0))
			top1.update(prec1[0], input.size(0))
			top5.update(prec5[0], input.size(0))

			# compute gradient and do SGD step
			optimizer.zero_grad()
			loss.backward()

			# Update the noise for the next iteration
			pert = step_size*torch.sign(noise_batch.grad)

			global_noise_data[0:input.size(0)] += pert.data
			global_noise_data.clamp_(-clip_eps,clip_eps)

			optimizer.step()
			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

		if i % 10 == 0:
			progress.display(i)

		# TODO: this should also be done with the helper.ProgressMeter
		print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
			  .format(top1=top1, top5=top5))

		return top1.avg, top5.avg,losses.avg, global_noise_data