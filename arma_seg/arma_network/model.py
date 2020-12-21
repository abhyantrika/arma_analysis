# reference: https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/master/network/modeling.py
# header files
from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3
from .fcn import FCNHead, FCN
from .backbone import resnet


def _segm_resnet(name, backbone_name, num_classes, output_stride, arma, pretrained_with_arma_backbone):
    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]
    backbone = resnet.__dict__[backbone_name](replace_stride_with_dilation=replace_stride_with_dilation, arma=arma, pretrained_with_arma=pretrained_with_arma_backbone)
    inplanes = 2048
    low_level_planes = 256

    if name=='deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate, arma=arma)
    elif name=='deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate, arma=arma)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model

def _segm_resnet_fcn(backbone_name, num_classes, output_stride, arma, pretrained_with_arma_backbone):
    replace_stride_with_dilation=[False, False, False]
    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
    elif output_stride==16:
        replace_stride_with_dilation=[False, False, True]

    backbone = resnet.__dict__[backbone_name](replace_stride_with_dilation=replace_stride_with_dilation, arma=arma, pretrained_with_arma=pretrained_with_arma_backbone)
    return_layers = {'layer4': 'out'}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    classifier = FCNHead(512, num_classes, arma=arma)

    model = FCN(backbone, classifier)
    return model

def _load_model(arch_type, backbone, num_classes, output_stride, arma, pretrained_with_arma_backbone):
    if backbone=='mobilenetv2':
        model = _segm_mobilenet(arch_type, backbone, num_classes, output_stride=output_stride, arma=arma, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('resnet'):
        model = _segm_resnet(arch_type, backbone, num_classes, output_stride=output_stride, arma=arma, pretrained_with_arma_backbone=pretrained_with_arma_backbone)
    else:
        raise NotImplementedError
    return model

def _load_model_fcn(backbone, output_stride, num_classes, arma, pretrained_with_arma_backbone):
    if backbone=='mobilenetv2':
        model = _segm_mobilenet(arch_type, backbone, num_classes, output_stride=output_stride, arma=arma, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('resnet'):
        model = _segm_resnet_fcn(backbone_name=backbone, num_classes=num_classes, output_stride=output_stride, arma=arma, pretrained_with_arma_backbone=pretrained_with_arma_backbone)
    else:
        raise NotImplementedError
    return model

def _load_resnet(arch_type, num_classes, arma):
    backbone = resnet.__dict__[arch_type](replace_stride_with_dilation=[False, False, False], arma=arma, num_classes=num_classes)
    return backbone

def _load_model_fcn(backbone, output_stride, num_classes, arma, pretrained_with_arma_backbone):
    if backbone=='mobilenetv2':
        model = _segm_mobilenet(arch_type, backbone, num_classes, output_stride=output_stride, arma=arma, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('resnet'):
        model = _segm_resnet_fcn(backbone_name=backbone, num_classes=num_classes, output_stride=output_stride, arma=arma, pretrained_with_arma_backbone=pretrained_with_arma_backbone)
    else:
        raise NotImplementedError
    return model


# Deeplab v3
def deeplabv3_resnet18(num_classes=19, output_stride=8, arma=False, pretrained_with_arma_backbone=False):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        arma: boolean value
    """
    return _load_model('deeplabv3', 'resnet18', num_classes, output_stride=output_stride, arma=arma, pretrained_with_arma_backbone=pretrained_with_arma_backbone)

def deeplabv3_resnet50(num_classes=19, output_stride=8, arma=False, pretrained_with_arma_backbone=False):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        arma: boolean value
    """
    return _load_model('deeplabv3', 'resnet50', num_classes, output_stride=output_stride, arma=arma, pretrained_with_arma_backbone=pretrained_with_arma_backbone)

def deeplabv3_resnet101(num_classes=19, output_stride=8, arma=False, pretrained_with_arma_backbone=False):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        arma: boolean value
    """
    return _load_model('deeplabv3', 'resnet101', num_classes, output_stride=output_stride, arma=arma, pretrained_with_arma_backbone=pretrained_with_arma_backbone)


# Deeplab v3+
def deeplabv3plus_resnet18(num_classes=19, output_stride=8, arma=False, pretrained_with_arma_backbone=False):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        arma: boolean value
    """
    return _load_model('deeplabv3plus', 'resnet18', num_classes, output_stride=output_stride, arma=arma, pretrained_with_arma_backbone=pretrained_with_arma_backbone)

def deeplabv3plus_resnet50(num_classes=19, output_stride=8, arma=False, pretrained_with_arma_backbone=False):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        arma: boolean value
    """
    return _load_model('deeplabv3plus', 'resnet50', num_classes, output_stride=output_stride, arma=arma, pretrained_with_arma_backbone=pretrained_with_arma_backbone)

def deeplabv3plus_resnet101(num_classes=19, output_stride=8, arma=False, pretrained_with_arma_backbone=False):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        arma: boolean value
    """
    return _load_model('deeplabv3plus', 'resnet101', num_classes, output_stride=output_stride, arma=arma, pretrained_with_arma_backbone=pretrained_with_arma_backbone)


# FCN
def fcn_resnet18(num_classes=19, output_stride=32, arma=False, pretrained_with_arma_backbone=False):
    """Constructs a FCN model with a ResNet-18 backbone.
    Args:
        num_classes (int): number of classes.
        output_stride: the ratio of the size of input image to size of image at the end of the encoder
        arma: boolean value
        pretrained_with_arma_backbone: boolean indicating whether to use pretrained backbone or not
    """
    return _load_model_fcn('resnet18', output_stride=output_stride, num_classes=num_classes, arma=arma, pretrained_with_arma_backbone=pretrained_with_arma_backbone)

def fcn_resnet50(num_classes=19, output_stride=32, arma=False, pretrained_with_arma_backbone=False):
    """Constructs a FCN model with a ResNet-50 backbone.
    Args:
        num_classes (int): number of classes.
        output_stride: the ratio of the size of input image to size of image at the end of the encoder
        arma: boolean value
        pretrained_with_arma_backbone: boolean indicating whether to use pretrained backbone or not
    """
    return _load_model_fcn('resnet50', output_stride=output_stride, num_classes=num_classes, arma=arma, pretrained_with_arma_backbone=pretrained_with_arma_backbone)


# Backbones
def resnet18(num_classes=1000, arma=False):
    """Constructs a FCN model with a ResNet-18 backbone.
    Args:
        num_classes (int): number of classes.
        arma: boolean value
    """
    return _load_resnet('resnet18', num_classes=num_classes, arma=arma)

def resnet50(num_classes=1000, arma=False):
    """Constructs a FCN model with a ResNet-50 backbone.
    Args:
        num_classes (int): number of classes.
        arma: boolean value
    """
    return _load_resnet('resnet50', num_classes=num_classes, arma=arma)

def resnet101(num_classes=1000, arma=False):
    """Constructs a FCN model with a ResNet-101 backbone.
    Args:
        num_classes (int): number of classes.
        arma: boolean value
    """
    return _load_resnet('resnet101', num_classes=num_classes, arma=arma)
