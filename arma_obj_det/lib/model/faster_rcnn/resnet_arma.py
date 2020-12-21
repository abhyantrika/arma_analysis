from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
import torch.nn as nn
import torch.nn.functional as F
from .ARMA_Layer import ARMA2d


from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn import _fasterRCNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
       'resnet152']



model_with_arma_files = {
    'resnet18': '.pth',
    'resnet34': '.pth',
    'resnet50': '.pth',
    'resnet101': '.pth',
    'resnet152': '.pth',
}

model_without_arma_files = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, arma=True):
    """
        in_planes: number of input channels
        out_planes: number of output channels
        stride: the stride value
        groups: the number of groups
        dilation: the gap between kernel cells
        arma: True, then arma layer applied, otherwise conv layer
    """
    if arma:
      return ARMA2d(in_planes, out_planes, w_kernel_size=3, w_padding=dilation, w_stride=stride, w_groups=groups, w_dilation=dilation, bias=False)
    else:
      return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1, arma=True):
    """
        in_planes: number of input channels
        out_planes: number of output channels
        stride: the stride value
        arma: True, then arma layer applied, otherwise conv layer
    """
    if arma:
      return ARMA2d(in_planes, out_planes, w_kernel_size=1, w_stride=stride, w_padding=0, bias=False)
    else:
      return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None, arma=True):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation>1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride, arma=arma)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, arma=arma)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None, arma=True):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes*(base_width/64.))*groups
        
        self.conv1 = conv1x1(inplanes, width, arma=arma)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, arma=arma)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes*4, arma=arma)
        self.bn3 = norm_layer(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=21, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, arma=True):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        if(arma):
            self.conv1 = ARMA2d(3, self.inplanes, w_kernel_size=7, w_stride=2, w_padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], arma=arma)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], arma=arma)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], arma=arma)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], arma=arma)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, arma=True):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes*block.expansion, stride, arma=arma),
                norm_layer(planes*block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, arma=arma))
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer, arma=arma))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def _resnet(arch, block, layers, arma=True, pretrained_with_arma=True, **kwargs):
    # load resnet
    model = ResNet(block, layers, arma=arma, **kwargs)

    # if pretrained with arma
    # if pretrained_with_arma and arma:
    #     model.load_state_dict(model_with_arma_files[arch])

    # # if pretrained without arma
    # if arma == False:
    #     model.load_state_dict(model_zoo.load_url(model_without_arma_files[arch]))

    return model


def resnet18(arma=True, pretrained_with_arma=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], arma=arma, pretrained_with_arma=pretrained_with_arma, **kwargs)


class resnet(_fasterRCNN):
  def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False, pretrained_path =''):
    self.model_path = 'data/pretrained_model/resnet101_caffe.pth'
    if num_layers!=18:
      self.dout_base_model = 1024
    else:
      self.dout_base_model = 256
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    self.num_layers = num_layers
    self.pretrained_path = pretrained_path

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    if self.num_layers == 101:
      resnet = resnet101(self.pretrained)
    elif self.num_layers == 50:
      resnet = resnet50(self.pretrained)
    if self.num_layers == 18:
      resnet = resnet18(arma=True,pretrained_with_arma=False)

    if self.pretrained == True and self.num_layers == 101:
      print("Loading pretrained weights from %s" %(self.model_path))
      state_dict = torch.load(self.model_path)
      resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})

    if self.pretrained_path:
      checkpoint = torch.load(self.pretrained_path)
      resnet.load_state_dict({'.'.join(k.split('.')[1:]):v for k,v in checkpoint["state_dict"].items()})
      # cfg.RESNET.FIXED_BLOCKS = 3

    # Build resnet.
    self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
      resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)

    self.RCNN_top = nn.Sequential(resnet.layer4)

    in_dim = 512 if self.num_layers==18 else 2048
    self.RCNN_cls_score = nn.Linear(in_dim, self.n_classes)
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(in_dim, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(in_dim, 4 * self.n_classes)

    # Fix blocks
    for p in self.RCNN_base[0].parameters(): p.requires_grad=False
    for p in self.RCNN_base[1].parameters(): p.requires_grad=False

    assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    if cfg.RESNET.FIXED_BLOCKS >= 3:
      for p in self.RCNN_base[6].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 2:
      for p in self.RCNN_base[5].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 1:
      for p in self.RCNN_base[4].parameters(): p.requires_grad=False

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.RCNN_base.apply(set_bn_fix)
    self.RCNN_top.apply(set_bn_fix)

  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    # if mode:
    #   # Set fixed blocks to be in eval mode
    #   self.RCNN_base.eval()
    #   self.RCNN_base[5].train()
    #   self.RCNN_base[6].train()

    #   def set_bn_eval(m):
    #     classname = m.__class__.__name__
    #     if classname.find('BatchNorm') != -1:
    #       m.eval()

    #   self.RCNN_base.apply(set_bn_eval)
    #   self.RCNN_top.apply(set_bn_eval)

  def _head_to_tail(self, pool5):
    fc7 = self.RCNN_top(pool5).mean(3).mean(2)
    return fc7





# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, w_ksz=3, a_ksz=3, stride=1, init=0,arma=True):
#         super(BasicBlock, self).__init__()
#         if arma:
#             self.conv1 = ARMA2d(in_planes, planes, w_stride=stride, w_kernel_size=w_ksz, w_padding=w_ksz//2, 
#                                                    a_init=init, a_kernel_size=a_ksz, a_padding=a_ksz//2)
#         else:
#             self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=w_ksz, stride=stride, padding=w_ksz//2, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)

#         if arma:
#             self.conv2 =  ARMA2d(planes, planes, w_kernel_size=w_ksz, w_padding=w_ksz//2, 
#                                                  a_init=init, a_kernel_size=a_ksz, a_padding=a_ksz//2)
#         else:
#             self.conv2 = nn.Conv2d(planes, planes, kernel_size=w_ksz, stride=1, padding=w_ksz//2, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             if arma:
#                 self.shortcut = nn.Sequential(
#                     ARMA2d(in_planes, self.expansion*planes, w_kernel_size=1, w_stride=stride, w_padding=0, 
#                                                              a_init=init, a_kernel_size=a_ksz, a_padding=a_ksz//2),
#                     nn.BatchNorm2d(self.expansion*planes)
#                 )
#             else:
#                 self.shortcut = nn.Sequential(
#                     nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                     nn.BatchNorm2d(self.expansion*planes)
#                 )

#     def forward(self, x):
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = self.relu(out)
#         return out


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, in_planes, planes, arma=True, w_ksz=3, a_ksz=3, stride=1, init=0):
#         super(Bottleneck, self).__init__()
#         if arma:
#             self.conv1 = ARMA2d(in_planes, planes, w_kernel_size=1, w_padding=0, 
#                                                    a_init=init, a_kernel_size=a_ksz, a_padding=a_ksz)
#         else:
#             self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)

#         if arma:
#             self.conv2 =ARMA2d(planes, planes,w_stride=stride, w_kernel_size=w_ksz, w_padding=w_ksz//2,
#                                               a_init=init, a_kernel_size=a_ksz, a_padding=a_ksz//2)
#         else:
#             self.conv2 = nn.Conv2d(planes, planes, kernel_size=w_ksz, stride=stride, padding=w_ksz//2, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         if arma:
#             self.conv3 = ARMA2d(planes, self.expansion*planes, w_kernel_size=1, w_padding=0, a_init=init)
#         else:
#             self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.expansion*planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             if arma:
#                 self.shortcut = nn.Sequential(
#                     ARMA2d(in_planes, self.expansion*planes, w_kernel_size=1, w_stride=stride, w_padding=0, 
#                            a_init=init),
#                     nn.BatchNorm2d(self.expansion*planes)
#                 )
#             else:
#                 self.shortcut = nn.Sequential(
#                     nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                     nn.BatchNorm2d(self.expansion*planes)
#                 )

#     def forward(self, x):
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = self.relu(out)
#         return out


# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, arma, num_classes, init=0, w_ksz=3, a_ksz=3):
#         super(ResNet, self).__init__()
#         self.in_planes = 64

#         if arma:
#             self.conv1 = ARMA2d(3, 64, w_kernel_size=w_ksz, w_padding=w_ksz//2, 
#                                        a_init=init, a_kernel_size=a_ksz, a_padding=a_ksz//2)
#         else:
#             self.conv1 = nn.Conv2d(3, 64, kernel_size=w_ksz, stride=1, padding=w_ksz//2, bias=False)
        
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block,  64, num_blocks[0], arma, w_ksz, a_ksz, 
#                                                                  stride=1, init=init)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], arma, w_ksz, a_ksz, 
#                                                                  stride=2, init=init)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], arma, w_ksz, a_ksz, 
#                                                                  stride=2, init=init)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], arma, w_ksz, a_ksz, 
#                                                                  stride=2, init=init)
#         self.linear = nn.Linear(512*block.expansion, num_classes)
#         self.softmax = nn.LogSoftmax(dim=-1)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
#         self.relu = nn.ReLU(inplace=True)

#     def _make_layer(self, block, planes, num_blocks, arma, w_ksz, a_ksz, stride, init):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, arma, w_ksz, a_ksz, stride, init))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = self.relu(self.bn1(self.conv1(x)))
#         breakpoint()
#         out = self.maxpool(out)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)

#         #out = F.avg_pool2d(out, 4)
#         out = F.adaptive_avg_pool2d(out,1)
        
#         out = out.view(out.size(0), -1)
#         #print(out.shape)
#         out = self.linear(out)
#         #out = self.softmax(out)
#         return out

# def resnet18(pretrained=False,arma=True,num_classes=21):
#   """Constructs a ResNet-18 model.
#   Args:
#     pretrained (bool): If True, returns a model pre-trained on ImageNet
#   """
#   model = ResNet(BasicBlock, [2, 2, 2, 2],arma=arma,num_classes=num_classes)

#   #ResNet(block, num_blocks, arma, num_classes, rf_init, w_kernel_size, a_kernel_size)
#   # if pretrained:
#   #   model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
#   return model

# class resnet(_fasterRCNN):
#   def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False, pretrained_path ='',arma=True):
#     self.model_path = 'data/pretrained_model/resnet101_caffe.pth'
#     if num_layers!=18:
#       self.dout_base_model = 1024
#     else:
#       self.dout_base_model = 256
#     self.pretrained = pretrained
#     self.class_agnostic = class_agnostic
#     self.num_layers = num_layers
#     self.pretrained_path = pretrained_path

#     self.arma = arma
#     self.classes = classes
#     self.num_classes = len(self.classes)

#     _fasterRCNN.__init__(self, classes, class_agnostic)

#   def _init_modules(self):
#     if self.num_layers == 101:
#       resnet = resnet101(self.pretrained)
#     elif self.num_layers == 50:
#       resnet = resnet50(self.pretrained)
#     if self.num_layers == 18:
#       resnet = resnet18(self.pretrained,self.arma,self.num_classes)

#     if self.pretrained == True and self.num_layers == 101:
#       print("Loading pretrained weights from %s" %(self.model_path))
#       state_dict = torch.load(self.model_path)
#       resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})

#     if self.pretrained_path:
#       checkpoint = torch.load(self.pretrained_path)
#       resnet.load_state_dict({'.'.join(k.split('.')[1:]):v for k,v in checkpoint["state_dict"].items()})
#       # cfg.RESNET.FIXED_BLOCKS = 3

#     # Build resnet.
    
#     self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
#       resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)

#     self.RCNN_top = nn.Sequential(resnet.layer4)

#     in_dim = 512 if self.num_layers==18 else 2048
#     self.RCNN_cls_score = nn.Linear(in_dim, self.n_classes)
#     if self.class_agnostic:
#       self.RCNN_bbox_pred = nn.Linear(in_dim, 4)
#     else:
#       self.RCNN_bbox_pred = nn.Linear(in_dim, 4 * self.n_classes)

#     # Fix blocks
#     for p in self.RCNN_base[0].parameters(): p.requires_grad=False
#     for p in self.RCNN_base[1].parameters(): p.requires_grad=False

#     assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
#     if cfg.RESNET.FIXED_BLOCKS >= 3:
#       for p in self.RCNN_base[6].parameters(): p.requires_grad=False
#     if cfg.RESNET.FIXED_BLOCKS >= 2:
#       for p in self.RCNN_base[5].parameters(): p.requires_grad=False
#     if cfg.RESNET.FIXED_BLOCKS >= 1:
#       for p in self.RCNN_base[4].parameters(): p.requires_grad=False

#     def set_bn_fix(m):
#       classname = m.__class__.__name__
#       if classname.find('BatchNorm') != -1:
#         for p in m.parameters(): p.requires_grad=False

#     self.RCNN_base.apply(set_bn_fix)
#     self.RCNN_top.apply(set_bn_fix)

#   def train(self, mode=True):
#     # Override train so that the training mode is set as we want
#     nn.Module.train(self, mode)

#     #Train everything

#     # if mode:
#     #   # Set fixed blocks to be in eval mode
#     #   self.RCNN_base.eval()
#     #   self.RCNN_base[5].train()
#     #   self.RCNN_base[6].train()

#     #   def set_bn_eval(m):
#     #     classname = m.__class__.__name__
#     #     if classname.find('BatchNorm') != -1:
#     #       m.eval()

#     #   self.RCNN_base.apply(set_bn_eval)
#     #   self.RCNN_top.apply(set_bn_eval)

#   def _head_to_tail(self, pool5):
#     fc7 = self.RCNN_top(pool5).mean(3).mean(2)
#     return fc7



# # def ResNet_(model_arch="ResNet18", arma=True, dataset="CIFAR10", rf_init=0,
# #             w_kernel_size=3, a_kernel_size=3):
# #     num_classes       = {   "MNIST":  10,
# #                           "CIFAR10":  10,
# #                          "CIFAR100": 100,
# #                          "ImageNet":1000 }[dataset]

# #     block, num_blocks = { "ResNet18": (BasicBlock, [ 2,2,2,2]),
# #                           "ResNet34": (BasicBlock, [ 3,4,6,3]),
# #                           "ResNet50": (Bottleneck, [ 3,4,6,3]),
# #                          "ResNet101": (Bottleneck, [3,4,23,3]),
# #                          "ResNet152": (Bottleneck, [3,8,36,3]) }[model_arch]

# #     return ResNet(block, num_blocks, arma, num_classes, rf_init, w_kernel_size, a_kernel_size)


# if __name__ == '__main__':
#     import torch 
#     inp = torch.rand(2,3,224,224)

#     model = ResNet_("ResNet18",True, "ImageNet",0,3,3)

#     out = model(inp)

