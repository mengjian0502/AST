import torch.nn as nn
import torch.nn.functional as F
from .sparsmodule import SparsConv2d, SparsLinear
from .nm_sparsmodule import NMSparsConv2d, NMSparsLinear
from .usbn import SwitchBatchNorm2d

__all__ = ['resnet18_imagenet', 'resnet50_imagenet']

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, nspars=4, uspars=True):
        super(BasicBlock, self).__init__()

        self.conv1 = SparsConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, nspars=nspars, uspars=uspars)
        self.bn1 = SwitchBatchNorm2d(planes, nspars)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = SparsConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, nspars=nspars, uspars=uspars)
        self.bn2 = SwitchBatchNorm2d(planes, nspars)
        self.relu2 = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                SparsConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, nspars=nspars, uspars=uspars),
                SwitchBatchNorm2d(self.expansion*planes, nspars)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(self.bn1(out))
        
        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu2(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, nspars=4, uspars=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, nspars=4, uspars=True):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.nspars = nspars
        self.uspars = uspars

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu0 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, SparsConv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            # elif isinstance(m, (SwitchBatchNorm2d, nn.GroupNorm)):
            #     for n in range(nspars):
            #         m._switch(n)
            #         nn.init.constant_(m.bn[m.idx].weight, 1)
            #         nn.init.constant_(m.bn[m.idx].bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        # for m in self.modules():
        #     if isinstance(m, Bottleneck):
        #         for n in range(nspars):
        #             m.bn3._switch(n)
        #             nn.init.constant_(m.bn3.bn[n].weight, 0)  # type: ignore[arg-type]
        #     elif isinstance(m, BasicBlock):
        #         for n in range(nspars):
        #             m.bn2._switch(n)
        #             nn.init.constant_(m.bn2.bn[n].weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def switch(self, n):
        for m in self.modules():
            if isinstance(m, (SparsConv2d, SparsLinear, SwitchBatchNorm2d)):
                m._switch(n)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu0(self.bn1(out))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class resnet18_imagenet:
    base = ResNet
    args = list()
    kwargs = {'block': BasicBlock, 'num_blocks': [2, 2, 2, 2]}

class resnet50_imagenet:
    base = ResNet
    args = list()
    kwargs = {'block': Bottleneck, 'num_blocks': [3, 4, 6, 3]}