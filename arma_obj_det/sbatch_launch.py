import os
from datetime import datetime
import argparse
import time
import socket
import itertools

# Function to chec for validity of QOS
#TODO: Add time check for QOS

qos_dict = {
            "scav" : {"nhrs" : 72, "cores": 16, "mem":128},
            "high" : {"gpu":4, "cores": 16, "mem":128, "nhrs": 36},
            "medium" : {"gpu":2, "cores": 8, "mem":64, "nhrs": 72},
            "default" : {"gpu":1, "cores": 4, "mem":32, "nhrs": 168}}


def check_qos(args):
    
    for key, max_value in qos_dict[args.qos].items():
        val_from_args = getattr(args, key)
        if val_from_args != None:
            if val_from_args > max_value:
                raise ValueError("Invalid paramter for {} for {}".format(key, args.qos))
        else:
            setattr(args, key, max_value)
    return args


#TODO: Add day funtionality too 
parser = argparse.ArgumentParser()
parser.add_argument('--nhrs', type=int, default=None)
parser.add_argument('--base-dir', default=f'{os.getcwd()}')
parser.add_argument('--output-dirname', default='outputs')
parser.add_argument('--dryrun', action='store_true')
parser.add_argument('--qos', default=None, type=str, help='Qos to run')
parser.add_argument('--env', type=str, help = "Set the name of the dir you want to dump")
parser.add_argument('--gpu', default=None, type=int, help='Number of gpus')
parser.add_argument('--cores', default=None, type=int, help='Number of cpu cores')
parser.add_argument('--mem', default=None, type=int, help='RAM in G')





args = parser.parse_args()

output_dir = os.path.join(args.base_dir, args.output_dirname, args.env)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print("Output Directory: %s" % output_dir)


#params_to_iterate
params = [(max_epoch, batch_size, lr, lr_decay_step, keep_percentage, num_rounds, late_reset_iter, lr_warmup, random_seed)
                            for max_epoch in [12]
                            for batch_size in [12]
                            for lr in [1e-2]
                            for lr_decay_step in [10]
                            for keep_percentage in [0.1]
                            for num_rounds in [2,3,4]
                            for late_reset_iter in [0]
                            for lr_warmup in [0]
                            for random_seed in [422,3423,632,7564,9287]]


#######################################################################
os.makedirs(f'{args.base_dir}/slurm_files/{args.env}',exist_ok=True)
# Making text files which will store the python command to run, stdout, and error if any  
with open(f'{args.base_dir}/slurm_files/{args.env}/now.txt', "w") as nowfile,\
     open(f'{args.base_dir}/slurm_files/{args.env}/log.txt', "w") as output_namefile,\
     open(f'{args.base_dir}/slurm_files/{args.env}/err.txt', "w") as error_namefile,\
     open(f'{args.base_dir}/slurm_files/{args.env}/name.txt', "w") as namefile:

    for i, (max_epoch, batch_size, lr, lr_decay_step, keep_percentage, num_rounds, late_reset_iter, lr_warmup, random_seed) in enumerate(params):
        print(i)
        now = datetime.now()
        datetimestr = now.strftime("%m%d_%H%M:%S.%f")
        name = f'training_{args.env}_e{max_epoch}_bs{batch_size}_lr{lr}_lrds{lr_decay_step}_kp{keep_percentage}'+\
               f'_nr{num_rounds}_lri{late_reset_iter}_lrw{lr_warmup}_rs{random_seed}'

        #python comand to run 
        cmd  = f'python trainval_net.py --dataset pascal_voc --net res18 --cuda --mGPUs --epochs {max_epoch} --bs {batch_size} --lr {lr} '+\
               f'--lr_decay_step {lr_decay_step} --kp {keep_percentage} --nr {num_rounds} --lri {late_reset_iter} --lr_warmup {lr_warmup} --rs {random_seed}'

        nowfile.write(f'{cmd}\n')
        namefile.write(f'{(os.path.join(output_dir, name))}.log\n')
        output_namefile.write(f'{(os.path.join(output_dir, name))}_log.txt\n')
        error_namefile.write(f'{(os.path.join(output_dir, name))}_error.txt\n')
###########################################################################
# Make a {name}.slurm file in the {output_dir} which defines this job.
#slurm_script_path = os.path.join(output_dir, '%s.slurm' % name)
start=1
slurm_script_path = os.path.join(output_dir, f'{args.qos[:2]}_r0.slurm')
slurm_command = "sbatch %s" % slurm_script_path

# Make the .slurm file
with open(slurm_script_path, 'w') as slurmfile:
    slurmfile.write("#!/bin/bash\n")
    slurmfile.write(f"#SBATCH --array=1-{len(params)}\n")
    slurmfile.write("#SBATCH --output=/dev/null\n")
    slurmfile.write("#SBATCH --error=/dev/null\n")
    slurmfile.write("#SBATCH --requeue\n")
    args = check_qos(args)

    if args.qos == "scav":
        slurmfile.write("#SBATCH --account=scavenger\n")
        slurmfile.write("#SBATCH --partition scavenger\n")
        slurmfile.write("#SBATCH --time=%d:00:00\n" % args.nhrs)
        slurmfile.write("#SBATCH --cpus-per-task=%d\n" % args.cores)
        slurmfile.write("#SBATCH --mem=%dG\n" % args.mem)
        if not args.gpu is None: 
            slurmfile.write("#SBATCH --gres=gpu:%d\n" % args.gpu)
        else:
            raise ValueError("Specify the gpus for scavenger")
    else:
        if "cml" in socket.gethostname():
            slurmfile.write("#SBATCH --account=abhinav\n")
        slurmfile.write("#SBATCH --qos=%s\n" % args.qos)
        slurmfile.write("#SBATCH --time=%d:00:00\n" % args.nhrs)
        slurmfile.write("#SBATCH --gres=gpu:%d\n" % args.gpu)
        slurmfile.write("#SBATCH --cpus-per-task=%d\n" % args.cores)
        slurmfile.write("#SBATCH --mem=%dG\n" % args.mem)

        
    slurmfile.write("\n")
    # slurmfile.write("cd " + args.base_dir + '\n')
    # slurmfile.write("eval \"$(conda shell.bash hook)\"" '\n')
    
    slurmfile.write("prune\n")
    slurmfile.write("env_prune\n")

    # slurmfile.write("mkdir -p /scratch0/pulkit/data/dataset_v1/known_classes \n")
    # if "cml" in socket.gethostname():
    #     slurmfile.write("~/msrsync/msrsync -P -p 16 /cmlscratch/pulkit/dataset/dataset_v1/known_classes/ /scratch0/pulkit/data/dataset_v1/known_classes \n")

    # else:
    #     slurmfile.write("~/msrsync/msrsync -P -p 16 /vulcan/scratch/abhinav/sailon_data/datasets_for_ta2/dataset_v0/image_data/dataset_v1/known_classes/ /scratch0/pulkit/data/dataset_v1/known_classes \n")

    # # slurmfile.write(f"srun --output=$(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/output/{args.env}/ouput.txt | tail -n 1) $(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/output/{args.env}/jobs.txt | tail -n 1)\n")
    slurmfile.write(f"srun --output=$(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/slurm_files/{args.env}/log.txt | tail -n 1) --error=$(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/slurm_files/{args.env}/err.txt | tail -n 1)  $(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/slurm_files/{args.env}/now.txt | tail -n 1)\n")
    # slurmfile.write("rm -rf /scratch0/pulkit/ \n")
    slurmfile.write("\n")
print(slurm_command)
print("Running on {}, with {} gpus, {} cores, {} mem for {} hour".format(args.qos, args.gpu, args.cores, args.mem , args.nhrs))
if not args.dryrun:
   os.system("%s &" % slurm_command)
