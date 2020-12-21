import os 
import glob 
import sys 
import args 



args = args.get_args()

save_folder = args.save_folder.strip('/')

baseline_save_folder = save_folder + '_baseline'
arma_save_folder = save_folder +'_arma'

arma_path = args.arma_path
baseline_path = args.baseline_path

model_arch = args.model_arch

epsilons = [0.0078,0.015,0.031,0.06,0.12]

#correction. Run again.
#epsilons = [0.0078]

attacks = ['fgsm']

if not os.path.exists(save_folder):
	os.mkdir(save_folder)
if not os.path.exists(baseline_save_folder):
	os.mkdir(baseline_save_folder)
if not os.path.exists(arma_save_folder):	
	os.mkdir(arma_save_folder)

for eps in epsilons:
	for att in attacks:

		baseline_attack_cmd = 'python attack.py --model_path ' + baseline_path + ' --model_arch '+args.model_arch+ \
		' --eps '+str(eps) + ' --batch_size 256 --baseline --save_folder ' + \
		baseline_save_folder  + ' --attack '+str(att) +' --iters 10'

		

		arma_attack_cmd = 'python attack.py --model_path ' + arma_path + ' --model_arch ' + args.model_arch +\
		' --eps '+str(eps) + ' --batch_size 256 ' + '--save_folder ' + arma_save_folder +' --attack '+\
		str(att) +' --iters 10'
		
		print(baseline_attack_cmd)
		print(arma_attack_cmd)

		os.system(baseline_attack_cmd)
		os.system(arma_attack_cmd)
