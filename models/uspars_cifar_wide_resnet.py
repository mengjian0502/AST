import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .sparsmodule import SparsConv2d, SparsLinear
from .usbn import SwitchBatchNorm2d

_AFFINE = True


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, nspars=4, uspars=True):
        super(BasicBlock, self).__init__()
        self.conv1 = SparsConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, nspars=nspars, uspars=uspars)
        self.bn1 = SwitchBatchNorm2d(planes, nspars)
        self.conv2 = SparsConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, nspars=nspars, uspars=uspars)
        self.bn2 = SwitchBatchNorm2d(planes, nspars)

        self.downsample = None
        self.bn3 = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                SparsConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False, nspars=nspars, uspars=uspars))
            self.bn3 = SwitchBatchNorm2d(planes, nspars)

    def forward(self, x):
        # x: batch_size * in_c * h * w
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.bn3(self.downsample(x))
        out += residual
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, nspars=4, uspars=True):
        super(ResNet, self).__init__()
        _outputs = [32, 64, 128]
        self.in_planes = _outputs[0]

        self.conv1 = nn.Conv2d(3, _outputs[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(_outputs[0], affine=_AFFINE)
        self.layer1 = self._make_layer(block, _outputs[0], num_blocks[0], stride=1, nspars=nspars, uspars=uspars)
        self.layer2 = self._make_layer(block, _outputs[1], num_blocks[1], stride=2, nspars=nspars, uspars=uspars)
        self.layer3 = self._make_layer(block, _outputs[2], num_blocks[2], stride=2, nspars=nspars, uspars=uspars)
        self.linear = SparsLinear(_outputs[2], num_classes, nspars=nspars, uspars=uspars)

        self.apply(weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, nspars=4, uspars=True):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, nspars=nspars, uspars=uspars))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def switch(self, n):
        for m in self.modules():
            if isinstance(m, (SparsConv2d, SparsLinear, SwitchBatchNorm2d)):
                m._switch(n)


def resnet(depth=32, dataset='cifar10'):
    assert (depth - 2) % 6 == 0, 'Depth must be = 6n + 2, got %d' % depth
    n = (depth - 2) // 6
    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'cifar100':
        num_classes = 100
    elif dataset == 'tiny_imagenet':
        num_classes = 200
    else:
        raise NotImplementedError('Dataset [%s] is not supported.' % dataset)
    return ResNet(BasicBlock, [n]*3, num_classes)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


def weights_init(m):
    # print('=> weights init')
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # nn.init.normal_(m.weight, 0, 0.1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        # nn.init.xavier_normal(m.weight)
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        # Note that BN's running_var/mean are
        # already initialized to 1 and 0 respectively.
        if m.weight is not None:
            m.weight.data.fill_(1.0)
        if m.bias is not None:
            m.bias.data.zero_()

class uwresnet32:
    base=ResNet
    n = (32 - 2) // 6
    args = list()
    kwargs = {'block': BasicBlock, 'num_blocks':[n]*3}
    