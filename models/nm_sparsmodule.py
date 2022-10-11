"""
N:M sparsity
"""
from torch import autograd
import torch.nn.functional as F
from .sparsmodule import SparsConv2d, SparsLinear


class Sparse(autograd.Function):
    """" Prune the unimprotant edges for the forwards phase but pass the gradient to dense weight using STE in the backwards phase"""

    @staticmethod
    def forward(ctx, weight, mask, decay = 0.0002):

        ctx.save_for_backward(weight)
        output = weight.clone()
        ctx.mask = mask
        ctx.decay = decay

        return output*mask

    @staticmethod
    def backward(ctx, grad_output):

        weight, = ctx.saved_tensors
        return grad_output + ctx.decay * (1-ctx.mask) * weight, None, None

class NMSparsConv2d(SparsConv2d):    
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, dilation=1, groups: int = 1, bias: bool = True, nspars: int = 4, uspars: bool = False):
        super(NMSparsConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, nspars, uspars)


    def get_sparse_weights(self):
        return Sparse.apply(self.weight, self.mask)


    def forward(self, x):
        w = self.get_sparse_weights()
        x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

class NMSparsLinear(SparsLinear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, nspars: int = 4, uspars: bool = False):
        super(NMSparsLinear, self).__init__(in_features, out_features, bias, nspars, uspars)


    def get_sparse_weights(self):
        return Sparse.apply(self.weight, self.mask)

    def forward(self, x):
        w = self.get_sparse_weights()
        x = F.linear(x, w)
        return x