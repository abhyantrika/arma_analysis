#Currently FGSM
python attack_all.py --arma_path arma_resnet_18/checkpoints/0_model_best.pth --baseline_path\
 /vulcanscratch/shishira/lottery_stuff/LTH/experiments/baseline_cifar10/checkpoints/0_model_best.pth \
  --model_arch ResNet18  --save_folder attack_plots/resnet_18_attacks                       

 python attack_all.py --arma_path arma_vgg_11/checkpoints/0_model_best.pth   --baseline_path \
 /vulcanscratch/shishira/lottery_stuff/LTH/baseline_vgg_11/checkpoints/0_model_best.pth \
 --model_arch VGG11 --save_folder attack_plots/vgg_11_attacks
