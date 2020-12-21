import torch 
import numpy as np
import shutil
import math

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
			correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res


def adjust_learning_rate(optimizer, epoch, args):
	"""Decay the learning rate based on schedule"""
	lr = args.lr
	if args.cos:  # cosine lr schedule
		lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
	elif args.schedule is None: #Fixed lr.
		return 	
	else:  # stepwise lr schedule
		schedule = args.schedule 
		schedule = list(map(lambda x:int(x),schedule) )

		for milestone in schedule:
			lr *= 0.1 if epoch >= milestone else 1.
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

def check_create_dir(ckpt_dir):
	if not os.path.exists(ckpt_dir):
		os.mkdir(ckpt_dir)


def load_checkpoint(args,model,optimizer):

	""" 
		Checkpoint format:
				keys: state_dict,optimizer,saved_epoch,

	"""

	print("=> loading checkpoint '{}'".format(args.resume))
	
	if args.gpu is None:
		checkpoint = torch.load(args.resume)
	else:
		loc = 'cuda:{}'.format(args.gpu)
		checkpoint = torch.load(args.resume, map_location=loc)

	args.start_epoch = checkpoint['epoch']
	model.load_state_dict(checkpoint['state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer'])

	print("=> loaded checkpoint '{}' (epoch {})"
		  .format(args.resume, checkpoint['epoch']))


	return model,optimizer,args


def save_checkpoint(args,model,optimizer,epoch=0,is_best=False,periodic=False,custom_name=None):

	""" 
		Checkpoint format:
				keys: state_dict,optimizer,saved_epoch,
				Periodic: Save every epoch separately.
	"""

	if periodic:
		filename = args.exp_name + '/checkpoints/' + 'epoch_'+str(epoch)+'.pth'
	else:
		filename = args.exp_name + '/checkpoints/' + 'current.pth'

	if custom_name is not None:
		filename =  args.exp_name + '/checkpoints/' + custom_name

	
	state = {'epoch':epoch,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict()}
	torch.save(state, filename)

	if is_best:
		save_name = args.exp_name +'/checkpoints/' 
		shutil.copyfile(filename, save_name + 'model_best.pth.tar')



class runningScore(object):
	def __init__(self, n_classes):
		self.n_classes = n_classes
		self.confusion_matrix = np.zeros((n_classes, n_classes))

	def _fast_hist(self, label_true, label_pred, n_class):
		mask = (label_true >= 0) & (label_true < n_class)
		hist = np.bincount(
			n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
		).reshape(n_class, n_class)
		return hist

	def update(self, label_trues, label_preds):
		for lt, lp in zip(label_trues, label_preds):
			self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

	def get_scores(self):
		"""Returns accuracy score evaluation result.
			- overall accuracy
			- mean accuracy
			- mean IU
			- fwavacc
		"""
		hist = self.confusion_matrix
		acc = np.diag(hist).sum() / hist.sum()
		acc_cls = np.diag(hist) / hist.sum(axis=1)
		acc_cls = np.nanmean(acc_cls)
		iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
		mean_iu = np.nanmean(iu)
		freq = hist.sum(axis=1) / hist.sum()
		fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
		cls_iu = dict(zip(range(self.n_classes), iu))

		return (
			{
				"Overall Acc: \t": acc,
				"Mean Acc : \t": acc_cls,
				"FreqW Acc : \t": fwavacc,
				"Mean IoU : \t": mean_iu,
			},
			cls_iu,
		)

	def reset(self):
		self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))




from torch.optim.lr_scheduler import _LRScheduler


class ConstantLR(_LRScheduler):
	def __init__(self, optimizer, last_epoch=-1):
		super(ConstantLR, self).__init__(optimizer, last_epoch)

	def get_lr(self):
		return [base_lr for base_lr in self.base_lrs]


class PolynomialLR(_LRScheduler):
	def __init__(self, optimizer, max_iter, decay_iter=1, gamma=0.9, last_epoch=-1):
		self.decay_iter = decay_iter
		self.max_iter = max_iter
		self.gamma = gamma
		super(PolynomialLR, self).__init__(optimizer, last_epoch)

	def get_lr(self):
		if self.last_epoch % self.decay_iter or self.last_epoch % self.max_iter:
			return [base_lr for base_lr in self.base_lrs]
		else:
			factor = (1 - self.last_epoch / float(self.max_iter)) ** self.gamma
			return [base_lr * factor for base_lr in self.base_lrs]


class WarmUpLR(_LRScheduler):
	def __init__(
		self, optimizer, scheduler, mode="linear", warmup_iters=100, gamma=0.2, last_epoch=-1
	):
		self.mode = mode
		self.scheduler = scheduler
		self.warmup_iters = warmup_iters
		self.gamma = gamma
		super(WarmUpLR, self).__init__(optimizer, last_epoch)

	def get_lr(self):
		cold_lrs = self.scheduler.get_lr()

		if self.last_epoch < self.warmup_iters:
			if self.mode == "linear":
				alpha = self.last_epoch / float(self.warmup_iters)
				factor = self.gamma * (1 - alpha) + alpha

			elif self.mode == "constant":
				factor = self.gamma
			else:
				raise KeyError("WarmUp type {} not implemented".format(self.mode))

			return [factor * base_lr for base_lr in cold_lrs]

		return cold_lrs

