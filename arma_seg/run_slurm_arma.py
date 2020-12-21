import os
from datetime import datetime
import argparse
import time
import socket
import pdb
import numpy as np

# Function to chec for validity of QOS
#TODO: Add time check for QOS
def check_qos(args):
    qos_dict = {"high" : {"gpu":4, "cores": 16, "mem":128},
            "medium" : {"gpu":2, "cores": 8, "mem":64},
            "default" : {"gpu":1, "cores": 4, "mem":32}}
    for key, max_value in qos_dict[args.qos].items():
        val_from_args = getattr(args, key)
        if val_from_args != None:
            if val_from_args > max_value:
                raise ValueError("Invalid paramter for {} for {}".format(key, args.qos))
        else:
            setattr(args, key, max_value)
            print("Setting {} to max value of {} in QOS {} as not specified in arguements.".format(key, max_value, args.qos))
    return args


#TODO: Add day funtionality too 
parser = argparse.ArgumentParser()
parser.add_argument('--nhrs', type=int, default=72)
parser.add_argument('--base-dir', default=f'{os.getcwd()}')
parser.add_argument('--output-dirname', default='output')
parser.add_argument('--dryrun', action='store_true')
parser.add_argument('--scav', action='store_true')
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


#Setting the paramters for the scripts to run, modify as per your need
# params = [(lr, weight_decay,lr_scheme)
#             for lr in [0.03,0.01,0.001]
#             for weight_decay in [1e-4,5e-4]
#             for lr_scheme in ['fixed','cosine','schedule']
#           ]

params = [(lr, weight_decay,lr_scheme,mom)
            for lr in [0.1,0.03,0.01,0.05]
            for weight_decay in [5e-4]
            for lr_scheme in ['fixed','cosine','schedule']
            for mom in [0.99]
          ]


num_commands = len(params)

#######################################################################
# Making text files which will store the python command to run, stdout, and error if any  
with open(f'{args.base_dir}/output/{args.env}/now.txt', "w") as nowfile,\
     open(f'{args.base_dir}/output/{args.env}/log.txt', "w") as output_namefile,\
     open(f'{args.base_dir}/output/{args.env}/err.txt', "w") as error_namefile,\
     open(f'{args.base_dir}/output/{args.env}/name.txt', "w") as namefile:

     # Iterate over all hyper parameters
    for idx, ( lr, wd,lr_scheme,mom) in enumerate(params):
        now = datetime.now()
        datetimestr = now.strftime("%m%d_%H%M:%S.%f")
        name = f'{lr}_{wd}_{lr_scheme}'
        #cmd  = f'python dct_main.py --save_dir {args.env}_{idx} '
        # cmd = 'python main.py --dist-url tcp://localhost:10005 --multiprocessing-distributed --world-size 1 --rank 0 \
        #  --exp_name ' +'/output/'+ str(idx) +' --batch_size 32 --workers 16'

        cmd = 'python main_arma.py --exp_name ' +'output/'+ str(idx) +' --batch_size 32 --workers 16 \
        --data /scratch0/shishira/pascal_voc/ '


        cmd += f' --lr {lr}'
        cmd+= f' --momentum {mom}'
        cmd+= f' --weight-decay {wd}'
        cmd+= f' --epochs 350'

        if lr_scheme == 'cosine':
            cmd += f' --cos'
        elif lr_scheme =='schedule':
            cmd += f' --schedule 100 200'

        nowfile.write(f'{cmd}\n')
        namefile.write(f'{(os.path.join(output_dir, name))}.log\n')
        output_namefile.write(f'{(os.path.join(output_dir, name))}.log\n')
        error_namefile.write(f'{(os.path.join(output_dir, name))}.error\n')

        print(cmd)

###########################################################################
# Make a {name}.slurm file in the {output_dir} which defines this job.
#slurm_script_path = os.path.join(output_dir, '%s.slurm' % name)
start=1
slurm_script_path = os.path.join(output_dir, f'submit_{start}_{num_commands}.slurm')
slurm_command = "sbatch %s" % slurm_script_path

# Make the .slurm file
with open(slurm_script_path, 'w') as slurmfile:
    slurmfile.write("#!/bin/bash\n")
    slurmfile.write(f"#SBATCH --array=1-{num_commands}%20\n") #parallelize across commands.
    slurmfile.write("#SBATCH --output=/dev/null\n")
    slurmfile.write("#SBATCH --error=/dev/null\n")
    slurmfile.write("#SBATCH --requeue\n") #fuck. Restart the job 
    
    #slurmfile.write("#SBATCH --cpus-per-task=16\n")
    if args.scav:
        slurmfile.write("#SBATCH --account=scavenger\n")
        slurmfile.write("#SBATCH --partition scavenger\n")
        slurmfile.write("#SBATCH --time=%d:00:00\n" % args.nhrs)
        slurmfile.write("#SBATCH --gres=gpu:%d\n" % args.gpu)
        slurmfile.write("#SBATCH --cpus-per-task=%d\n" % args.cores)
        slurmfile.write("#SBATCH --mem=%dG\n" % args.mem)

    else:
        args = check_qos(args)
        slurmfile.write("#SBATCH --qos=%s\n" % args.qos)
        slurmfile.write("#SBATCH --time=%d:00:00\n" % args.nhrs)
        slurmfile.write("#SBATCH --gres=gpu:%d\n" % args.gpu)
        slurmfile.write("#SBATCH --cpus-per-task=%d\n" % args.cores)
        slurmfile.write("#SBATCH --mem=%dG\n" % args.mem)

        
    slurmfile.write("\n")
    slurmfile.write("cd " + args.base_dir + '\n')
    slurmfile.write("eval \"$(conda shell.zsh hook)\"" '\n')
    slurmfile.write("source /cmlscratch/shishira/miniconda3/bin/activate \n")

    slurmfile.write("mkdir -p /scratch0/shishira/ \n")

    slurmfile.write("/cmlscratch/shishira/msrsync pascal_voc /scratch0/shishira/ -p 20 -P \n")

    # if "cml" in socket.gethostname():
    #     slurmfile.write("~/msrsync/msrsync -P -p 16 /cmlscratch/pulkit/dataset/dataset_v1/known_classes/ /scratch0/pulkit/data/dataset_v1/known_classes \n")

    # else:
    #     slurmfile.write("~/msrsync/msrsync -P -p 16 /vulcan/scratch/abhinav/sailon_data/datasets_for_ta2/dataset_v0/image_data/dataset_v1/known_classes/ /scratch0/pulkit/data/dataset_v1/known_classes \n")

    # slurmfile.write(f"srun --output=$(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/output/{args.env}/ouput.txt | tail -n 1) $(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/output/{args.env}/jobs.txt | tail -n 1)\n")
    slurmfile.write(f"srun --output=$(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/output/{args.env}/log.txt | tail -n 1) --error=$(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/output/{args.env}/err.txt | tail -n 1)  $(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/output/{args.env}/now.txt | tail -n 1)\n")
    # slurmfile.write("rm -rf /scratch0/pulkit/ \n")
    slurmfile.write("\n")

print(slurm_command)
print("Running on {}, with {} gpus, {} cores, {} mem for {} hour".format(args.qos, args.gpu, args.cores, args.mem , args.nhrs))
if not args.dryrun:
   os.system("%s &" % slurm_command)
