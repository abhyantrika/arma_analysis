import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import os
from collections import  defaultdict
import torch
import numpy as np
import pdb

class Metric_counter(object):
	"""Class to store all metrics in one place.
		To store all freq RGB and all_freq y-channel for each image.
	"""

	def __init__(self,args):
		super(Metric_counter, self).__init__()

		self.per_freq_rgb = defaultdict(list)
		self.per_freq_y_channel = defaultdict(list)
		self.batch_wise_rgb = defaultdict(list)
		self.batch_wise_y_channel = defaultdict(list)
		
		#Total means for that freq accross images.
		self.rgb_means = []
		self.y_channel_means = []

		#Same for adversarial.
		self.per_freq_rgb_adv = defaultdict(list)
		self.per_freq_y_channel_adv = defaultdict(list)
		self.batch_wise_rgb_adv = defaultdict(list)
		self.batch_wise_y_channel_adv = defaultdict(list)

		#Total means for that freq accross images.
		self.rgb_means_adv = []
		self.y_channel_means_adv = []
		self.args = args

		self.batch_size = self.args.batch_size

		#Bandwise means. Dict with keys as each bands
		#self.rgb_band_mean = defaultdict(list)
		self.rgb_band_mean = []
		self.rgb_band_mean_adv = []
		self.y_band_mean = []
		self.y_band_mean_adv = []

	def rct(self,freq,adv_freq):

		"""	
			Naive normalization is useless. 
			Use relative DCT 
		"""	
		freq = np.array(freq)
		adv_freq = np.array(adv_freq)

		#rct = np.abs( (adv_freq - freq) / (freq ) )

		#pdb.set_trace()
		#rct = np.abs(adv_freq-freq) / (1+np.abs(freq))

		rct = []
		for i in range(len(adv_freq)):
			temp = np.abs(adv_freq[i] - freq[i])
			if freq[i]==0:
				rct.append(temp)
			else:
				rct.append( (temp/np.abs(freq[i])) )
		rct = np.array(rct)


		# if np.isinf(rct.mean()):
		# 	pdb.set_trace()
		#rct[rct==np.inf] = 0	
		#rct[rct==np.nan] = 0

		return rct 


	def add(self,rgb_freq,y_channel,rgb_freq_adv,y_channel_adv):
		#num_freqs is 64
		#For each image, get the mean values of all frequencies.
		batch_rgb = []
		batch_y = []
		for i in range(len(rgb_freq)):
			for f in range(64):
				#import pdb;
				#pdb.set_trace()
				self.per_freq_rgb[f].append(rgb_freq[i][f].mean().item())
				self.per_freq_y_channel[f].append(y_channel[i][f].mean().item())

				self.per_freq_rgb_adv[f].append(rgb_freq_adv[i][f].mean().item())
				self.per_freq_y_channel_adv[f].append(y_channel_adv[i][f].mean().item())

				# self.per_freq_rgb[f].append(self.normalize(rgb_freq[i][f]).mean().item())
				# self.per_freq_y_channel[f].append(self.normalize(y_channel[i][f]).mean().item())

				# self.per_freq_rgb_adv[f].append(self.normalize(rgb_freq_adv[i][f]).mean().item())
				# self.per_freq_y_channel_adv[f].append(self.normalize(y_channel_adv[i][f]).mean().item())



	def window_mean(self,temp,window_size):
		return [ np.mean(temp[i:i+window_size]) for i in range(0,len(temp),window_size)]

	def normalize(self,freq):
		"""
			Input takes in a single channel corresponding to a freq in the image.
			Normalizes the channel input to be between 0 and 1 and returns it.
		"""
		for ch in range(len(freq)):
			freq[ch] = (freq[ch] - freq[ch].min()) / (freq[ch].max() - freq[ch].min())
		
		return freq	

	def get_band_mean(self,freq_band):
		#reshape such that 4,N,c,h,w
		freq_band = freq_band.transpose(1,0)
		temp = [x.mean().item() for x in freq_band]
		return temp

	def add_bands(self,rgb_freq_bands,y_bands,rgb_freq_bands_adv,y_bands_adv):
		"""
			Plot the band means accross all images.
		"""

		#means = self.get_band_mean(rgb_freq_bands)
		means = rgb_freq_bands.mean([2,3,4])
		self.rgb_band_mean.append(means)
		self.rgb_band_mean_adv.append(rgb_freq_bands_adv.mean([2,3,4]))

		self.y_band_mean.append(y_bands.mean([2,3,4]))
		self.y_band_mean_adv.append(y_bands_adv.mean([2,3,4]))


	def plot_bands_adv(self,band_mean,band_mean_adv):
		"""	
			visualize band and the adv band in the same plot.
		"""
		pass


	def plot_bands(self,band_mean,filename):

		bandwise_folder = self.root_path +'/'+ 'bandwise_plots/'
		if not os.path.exists(bandwise_folder):
			os.mkdir(bandwise_folder)

		# try:
		# 	band_mean = torch.stack(band_mean)
		# except:
		# 	band_mean.pop(-1)
		# 	band_mean = torch.stack(band_mean)

		# n,x,_ = band_mean.shape
		# band_mean = band_mean.view(4,n*x)
		# band_mean = band_mean.cpu().detach().numpy()

		try:
			band_mean = torch.cat(band_mean)
		except:
			band_mean.pop(-1)
			band_mean = torch.cat(band_mean)

		band_mean = band_mean.cpu().detach().numpy()	

		plt.plot(band_mean[:,0],label='freq_0_15')
		plt.legend(loc='upper left')
		plt.title('epsilon='+str(self.args.eps))
		plt.savefig(bandwise_folder+filename+'_first.png')
		plt.clf()

		plt.plot(band_mean[:,1],label='freq_16_32')
		plt.plot(band_mean[:,2],label='freq_32_48')
		plt.plot(band_mean[:,3],label='freq_48_64')
		plt.legend(loc='upper left')
		plt.title('epsilon='+str(self.args.eps))

		plt.savefig(bandwise_folder+filename+'.png')		
		plt.clf()

	def create_batchwise(self):

		for k in self.per_freq_rgb.keys():

			temp = np.array(self.per_freq_rgb[k]).reshape(-1,1)
			temp = self.window_mean(self.per_freq_rgb[k],self.args.batch_size)
			self.batch_wise_rgb[k] += temp

			temp = np.array(self.per_freq_rgb_adv[k]).reshape(-1,1)
			temp = self.window_mean(self.per_freq_rgb_adv[k],self.args.batch_size)
			self.batch_wise_rgb_adv[k] += temp

			temp = np.array(self.per_freq_y_channel[k]).reshape(-1,1)
			temp = self.window_mean(self.per_freq_y_channel[k],self.args.batch_size)
			self.batch_wise_y_channel[k] += temp

			temp = np.array(self.per_freq_y_channel_adv[k]).reshape(-1,1)
			temp = self.window_mean(self.per_freq_y_channel_adv[k],self.args.batch_size)
			self.batch_wise_y_channel_adv[k] += temp


			self.rgb_means.append( np.mean(self.per_freq_rgb[k]) )
			self.y_channel_means.append( np.mean(self.per_freq_y_channel[k]) )

			self.rgb_means_adv.append( np.mean(self.per_freq_rgb_adv[k]) )
			self.y_channel_means_adv.append( np.mean(self.per_freq_y_channel_adv[k]) )



	def plot(self,root_path):
		self.root_path = root_path

		if not os.path.exists(self.root_path):
			os.mkdir(self.root_path)

		plt.plot(self.rgb_means,c='blue',label='og')
		plt.plot(self.rgb_means_adv,c='red',label='adv')
		plt.legend(loc='upper left')
		plt.title('epsilon='+str(self.args.eps))
		plt.savefig(self.root_path+'/rgb_means.png')
		plt.clf()

		plt.plot(self.rgb_means[1:],c='blue',label='og')
		plt.plot(self.rgb_means_adv[1:],c='red',label='adv')
		plt.legend(loc='upper left')
		plt.title('epsilon='+str(self.args.eps))
		plt.savefig(self.root_path+'/rgb_means_without_first.png')
		plt.clf()

		plt.plot(self.y_channel_means,c='blue',label='og')
		plt.plot(self.y_channel_means_adv,c='red',label='adv')
		plt.legend(loc='upper left')
		plt.title('epsilon='+str(self.args.eps))
		plt.savefig(self.root_path+'/y_channel_means.png')
		plt.clf()

		plt.plot(self.y_channel_means[1:],c='blue',label='og')
		plt.plot(self.y_channel_means_adv[1:],c='red',label='adv')
		plt.legend(loc='upper left')
		plt.title('epsilon='+str(self.args.eps))
		plt.savefig(self.root_path+'/y_channel_means_without_first.png')
		plt.clf()

		#pdb.set_trace()

		# #Plot differences.
		# diff_y = torch.dist(torch.tensor(self.y_channel_means[1:]),torch.tensor(self.y_channel_means_adv[1:]),2)
		# plt.plot(diff_y,c='blue',label='diff')
		# plt.title('epsilon='+str(self.args.eps))
		# plt.savefig(self.root_path+'/y_channel_means_diff_without_first.png')

		# diff_rgb = torch.dist(self.rgb_means[1:],self.rgb_means_adv[1:],2)
		# plt.plot(diff_rgb,c='blue',label='diff')
		# plt.title('epsilon='+str(self.args.eps))
		# plt.savefig(self.root_path+'/rgb_means_diff_without_first.png')


		batchwise_folder = root_path +'/'+ 'batchwise_plots'
		if not os.path.exists(batchwise_folder):
			os.mkdir(batchwise_folder)

		#Plot batchwise.
		#For each frequency
		for i in range(64):
			plt.plot(self.batch_wise_rgb[i],c='blue',label='og')
			plt.plot(self.batch_wise_rgb_adv[i],c='red',label='adv')
			plt.legend(loc='upper left')
			plt.title('epsilon='+str(self.args.eps))
			plt.savefig(batchwise_folder+'/'+'freq_'+str(i)+'_rgb.png')
			plt.clf()

			plt.plot(self.batch_wise_y_channel[i],c='blue',label='og')
			plt.plot(self.batch_wise_y_channel_adv[i],c='red',label='adv')
			plt.legend(loc='upper left')
			plt.title('epsilon='+str(self.args.eps))
			plt.savefig(batchwise_folder+'/'+'y_channel_'+str(i)+'_rgb.png')
			plt.clf()


		#Plot all samples for all freqencies except first one
		print("Per frequency plotted without first frequency.")
		for i in range(1,64):
			plt.plot(self.per_freq_rgb[i])
		plt.savefig(self.root_path+'/'+'per_freq_rgb.png')	
		plt.clf()

		for i in range(1,64):
			plt.plot(self.per_freq_rgb_adv[i])
		plt.savefig(self.root_path+'/'+'per_freq_rgb_adv.png')	
		plt.clf()

		for i in range(1,64):
			plt.plot(self.per_freq_y_channel[i])
		plt.savefig(self.root_path+'/'+'per_freq_y.png')	
		plt.clf()

		for i in range(1,64):
			plt.plot(self.per_freq_y_channel_adv[i])
		plt.savefig(self.root_path+'/'+'per_freq_y_channel_adv.png')	
		plt.clf()

		#Plot bandwise.
		self.plot_bands(self.rgb_band_mean,filename='bandwise_rgb_means')
		self.plot_bands(self.rgb_band_mean_adv,filename='bandwise_rgb_means_adv')

		self.plot_bands(self.y_band_mean,filename='bandwise_y_means')
		self.plot_bands(self.y_band_mean_adv,filename='bandwise_y_means_adv')

		#RCT plots
		#import pdb;
		#pdb.set_trace()
		self.rgb_rct = np.array([self.rct(x,y) for (x,y) in zip(self.per_freq_rgb.values(),self.per_freq_rgb_adv.values())])
		self.y_rct = np.array([self.rct(x,y) for (x,y) in zip(self.per_freq_y_channel.values(),self.per_freq_y_channel_adv.values())])


		#Mean them across all images, for all frequencies. 
		self.rgb_rct_mean = self.rgb_rct.mean(1)

		#pdb.set_trace()

		self.rgb_rct_mean[self.rgb_rct_mean==np.nan]=0
		#plt.plot(rgb_rct_mean)
		plt.bar(np.arange(64),self.rgb_rct_mean)

		if self.args.kernel_size is not None:
			plt.title('epsilon='+str(self.args.eps)+'_'+str(self.args.kernel_size) +'_'+str(self.args.sigma))
		else:
			plt.title('epsilon='+str(self.args.eps))

		#ylim = max(self.rgb_rct_mean) + 10
		ylim = 30
		plt.ylim([0,ylim])
		plt.savefig(self.root_path+'/'+'rgb_rct.png')
		plt.clf()

		self.y_rct_mean = self.y_rct.mean(1)
		#y_rct_mean[y_rct_mean==np.nan] = 0
		#plt.plot(y_rct_mean)
		plt.bar(np.arange(64),self.y_rct_mean)

		if self.args.kernel_size is not None:
			plt.title('epsilon='+str(self.args.eps)+'_'+str(self.args.kernel_size) +'_'+str(self.args.sigma))
		else:
			plt.title('epsilon='+str(self.args.eps))

		plt.ylim([0,20])
		plt.savefig(self.root_path+'/'+'y_rct.png')
		plt.clf()
