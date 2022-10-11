"""
MobileNetV1 on CIFAR-10
"""
import torch.nn as nn
from .sparsmodule import SparsConv2d, SparsLinear
from .usbn import SwitchBatchNorm2d

class Net(nn.Module):
    """
    Full precision mobilenet V1 model for CIFAR10
    """
    def __init__(self, alpha=1.0, num_classes=10, nspars=4, uspars=True):
        super(Net, self).__init__()
        self.alpha = alpha   # width multiplier of the model

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                SparsConv2d(inp, oup, 3, stride, 1, bias=False, nspars=nspars, uspars=uspars),
                SwitchBatchNorm2d(oup, nspars),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                SparsConv2d(inp, inp, 3, stride, 1, groups=inp, bias=False, nspars=nspars, uspars=uspars),
                SwitchBatchNorm2d(inp, nspars),
                nn.ReLU(inplace=True),

                SparsConv2d(inp, oup, 1, 1, 0, bias=False, nspars=nspars, uspars=uspars),
                SwitchBatchNorm2d(oup, nspars),
                nn.ReLU(inplace=True)
            )

        self.model = nn.Sequential(
            conv_bn(3, int(32*self.alpha), 1), 
            conv_dw(int(32*self.alpha),  int(64*self.alpha), 1),
            conv_dw(int(64*self.alpha), int(128*self.alpha), 2),
            conv_dw(int(128*self.alpha), int(128*self.alpha), 1),
            conv_dw(int(128*self.alpha), int(256*self.alpha), 2),
            conv_dw(int(256*self.alpha), int(256*self.alpha), 1),
            conv_dw(int(256*self.alpha), int(512*self.alpha), 2),
            conv_dw(int(512*self.alpha), int(512*self.alpha), 1),
            conv_dw(int(512*self.alpha), int(512*self.alpha), 1),
            conv_dw(int(512*self.alpha), int(512*self.alpha), 1),
            conv_dw(int(512*self.alpha), int(512*self.alpha), 1),
            conv_dw(int(512*self.alpha), int(512*self.alpha), 1),
            conv_dw(int(512*self.alpha), int(1024*self.alpha), 2),
            conv_dw(int(1024*self.alpha), int(1024*self.alpha), 1),
        )
        self.pool = nn.AvgPool2d(2)
        self.fc = SparsLinear(int(1024*self.alpha), num_classes, nspars=nspars, uspars=uspars)
    
    def switch(self, n):
        for m in self.modules():
            if isinstance(m, (SparsConv2d, SparsLinear, SwitchBatchNorm2d)):
                m._switch(n)

    def forward(self, x):
        x = self.model(x)
        x = self.pool(x)
        x = x.view(-1, int(1024*self.alpha))
        x = self.fc(x)
        return x

class umobilenetv1:
    base=Net
    args = list()
    kwargs = {'num_classes': 10}