import pandas as pd 
import os
import glob 
import re 
import pickle
import pickle
import pdb

def get_metric(metric_file,exp_dict,f):

	with open(metric_file,'rb') as fout:
		metric = pickle.load(fout)	


	scores,class_scores = metric.get_scores()	

	exp_dict[f].update({'total_acc':scores['Overall Acc: \t']})
	exp_dict[f].update({'mean_acc':scores['Mean Acc : \t']})
	exp_dict[f].update({'freqw_acc':scores['FreqW Acc : \t']})
	exp_dict[f].update({'mean_iou':scores['Mean IoU : \t']})

	exp_dict[f].update({'classwise_scores':class_scores})

	return exp_dict

folders = glob.glob('output/*')
exp_dict = {}

folders = [x for x in folders if '/out' not in x ]

for f in folders:
	files = glob.glob(f+'/*')

	params = pickle.load(open(f+'/args.pkl','rb'))
	metric_file = f+'/running_metric.pkl'

	exp_dict[f] = {'batch_size':params.batch_size,'cos':params.cos,'decay':params.weight_decay,\
	'epochs':params.epochs,'lr':params.lr,'momentum':params.momentum,'schedule':params.schedule}
	
	try:
		exp_dict = get_metric(metric_file,exp_dict,f)
	except:
		continue

	#break

df = pd.DataFrame.from_dict(exp_dict,orient='index')	
df.to_csv('results_arma_round_3_batch_32.csv')




