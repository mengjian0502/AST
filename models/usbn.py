"""
BatchNorm for sparse subnets
"""

import torch.nn as nn

class SwitchBatchNorm2d(nn.Module):
    def __init__(self, channels:int, nspars:int=4):
        super(SwitchBatchNorm2d, self).__init__()
        bn_list = []
        for i in range(nspars):
            bn_list.append(nn.BatchNorm2d(channels))
        self.bn = nn.ModuleList(bn_list)

        # bn index
        self.idx = 0

    def _switch(self, n):
        self.idx = n

    def forward(self, input):
        out = self.bn[self.idx](input)
        return out
