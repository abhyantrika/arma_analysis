# header files
import torch
from torch import nn
from torch.nn import functional as F

from .utils import _SimpleSegmentationModel
from .arma import *


__all__ = ["FCN"]


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels, arma=False):
        super(FCNHead, self).__init__()

        inter_channels = in_channels // 4
        if arma:
            self.classifier = nn.Sequential(
                ARMA2d(in_channels, inter_channels, w_kernel_size=3, w_padding=1, bias=False),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, channels, 1)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, channels, 1)
            )

    def forward(self, feature):
        return self.classifier(feature['out'])


class FCN(_SimpleSegmentationModel):
    """
    Implements FCN model
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass
