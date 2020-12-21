#python ARMA_CNN_train.py --use-cuda --multi-gpu --save-epoch 30 --learning-rate 0.1 --learning-rate-decay \
#--decay-epoch=30 --decay-rate=0.5  --epoch-num=300

#python baseline.py --outputs-path baseline/ --use-cuda --multi-gpu --save-epoch 30 --learning-rate 0.1\
# --learning-rate-decay --decay-epoch=30 --decay-rate=0.5  --epoch-num=300 --no-arma


python custom_train.py --exp_name arma_resnet_18 --lr 0.1 --epochs 350 --schedule 150 250 --model_arch ResNet18


python custom_train.py --exp_name arma_resnet_50 --lr 0.01 --epochs 350 --schedule 150 250 --model_arch ResNet50

