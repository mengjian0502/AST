"""
Light-weight resnet for CIFAR10
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .sparsmodule import SparsConv2d, SparsLinear
from .usbn import SwitchBatchNorm2d

class DownsampleA(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)
    
    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class ResNetBasicblock(nn.Module):
    expansion = 1
    """
    RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
    """
    def __init__(self, inplanes, planes, stride=1, downsample=None, nspars=4, uspars=True):
        super(ResNetBasicblock, self).__init__()


        self.conv_a = SparsConv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, nspars=nspars, uspars=uspars)
        self.bn_a = SwitchBatchNorm2d(planes, nspars)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv_b = SparsConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, nspars=nspars, uspars=uspars)
        self.bn_b = SwitchBatchNorm2d(planes, nspars)
        self.relu2 = nn.ReLU(inplace=True)    
        self.downsample = downsample

    def forward(self, x):
        residual = x

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = self.relu1(basicblock)

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        return self.relu2(residual + basicblock)


class CifarResNet(nn.Module):
    """
    ResNet optimized for the Cifar dataset, as specified in
    https://arxiv.org/abs/1512.03385.pdf
    """
    def __init__(self, depth, num_classes, nspars=4, uspars=True):
        """ Constructor
        Args:
        depth: number of layers.
        num_classes: number of classes
        base_width: base width
        """
        super(CifarResNet, self).__init__()

        block = ResNetBasicblock

        #Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6
        self.inflate = 1
        print ('CifarResNet : Depth : {} , Layers for each block : {}'.format(depth, layer_blocks))
        self.num_classes = num_classes
        self.conv_1_3x3 = nn.Conv2d(3, 16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)   # skip the push process for the first conv layer
        self.relu0 = nn.ReLU(inplace=True)
        self.bn_1 = nn.BatchNorm2d(16*self.inflate)

        self.inplanes = 16 * self.inflate 
        self.stage_1 = self._make_layer(block, 16*self.inflate, layer_blocks, 1, nspars=nspars, uspars=uspars)
        self.stage_2 = self._make_layer(block, 32*self.inflate, layer_blocks, 2, nspars=nspars, uspars=uspars)
        self.stage_3 = self._make_layer(block, 64*self.inflate, layer_blocks, 2, nspars=nspars, uspars=uspars)
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = SparsLinear(64*self.inflate, num_classes, nspars=nspars, uspars=uspars)                         # skip the push process for the last fc layer

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, nspars=4, uspars=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SparsConv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False, nspars=nspars, uspars=uspars),
                nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, nspars=nspars, uspars=uspars))
        self.inplanes = planes * block.expansion 
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, nspars=nspars, uspars=uspars))
        return nn.Sequential(*layers)

    def switch(self, n):
        for m in self.modules():
            if isinstance(m, (SparsConv2d, SparsLinear, SwitchBatchNorm2d)):
                m._switch(n)


    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = self.relu0(self.bn_1(x))
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class uresnet20:
    base=CifarResNet
    args = list()
    kwargs = {'depth': 20}

class uresnet32:
    base=CifarResNet
    args = list()
    kwargs = {'depth': 32}

class uresnet56:
    base=CifarResNet
    args = list()
    kwargs = {'depth': 56}
