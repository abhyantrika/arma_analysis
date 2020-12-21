#!/bin/bash

#SBATCH --requeue
#SBATCH --time=16:00:00
#SBATCH --account=scavenger
#SBATCH --partition scavenger
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8

cd /fs/vulcan-projects/pruning_sgirish/pytorch-segmentation-detection/my_lib/
source /cmlscratch/shishira/miniconda3/bin/activate 


mkdir -p /scratch0/shishira/
/vulcanscratch/shishira/msrsync pascal_voc /scratch0/shishira/ -p 20 -P

python main_normal.py --exp_name og_no_pre_train_batch_32 --cos --epochs 120 \
--lr 0.03 --momentum 0.9 --data /scratch0/shishira/pascal_voc/ --workers 16 --pre_train False



