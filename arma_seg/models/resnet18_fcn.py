import numpy as np
import torch.nn as nn
import torchvision.models as models

import sys 
sys.path.append('/fs/vulcan-projects/pruning_sgirish/pytorch-segmentation-detection/my_lib/models/')
import resnet

from torch.nn import functional as F

"""
Adapted from https://github.com/pytorch/vision/torchvision/models/segmentation/_utils.py#L8
and https://github.com/pytorch/vision/tree/1fe1e110677ab22f8512293987939d31916b7a8b/torchvision/models/segmentation

"""

class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]

        super(FCNHead, self).__init__(*layers)


class Resnet18_fcn(nn.Module):
    
    
    def __init__(self, num_classes=1000,pre_train=True):
        
        super(Resnet18_fcn, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        print('pretrained: ',pre_train)
        backbone = resnet.resnet18(fully_conv=True,
                                       pretrained=pre_train,
                                       output_stride=32,
                                       remove_avg_pool_layer=True,
                                       remove_fc=True)
 

        self.fcn = FCNHead(512,num_classes)
        
        self.backbone = backbone
        
        
    def forward(self, x):
        
        input_shape = x.shape[-2:]
        
        x = self.backbone(x)
        x = self.fcn(x)

        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        return x
    

if __name__ == '__main__':

    import torch
    inp = torch.rand(1,3,300,300)

    model = Resnet18_fcn(num_classes=21)

    out = model(inp)


