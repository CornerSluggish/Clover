import ctypes

lib = ctypes.cdll.LoadLibrary('./fully_con/fully_con.so')

import torch
from torch.autograd import Function
from torch.nn import Module
import torch.nn as nn
from torch.nn.parameter import Parameter
import math

import numpy as np


class Linear_4bit(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear_4bit, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'

    def forward(self, input):
        #return F.linear(input, self.weight, self.bias)
        return Mvm4Function()(input, self.weight, self.bias)


class Mvm4Function(Function):
    def forward(self, input, weight, bias=None):
        lib.mvm_4.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        lib.mvm_4.restype  = ctypes.POINTER(ctypes.c_float)

        in1 = ctypes.POINTER(ctypes.c_float)
        in2 = ctypes.POINTER(ctypes.c_float)

        fp1 = weight.detach().numpy().ctypes.data_as(in1)
        fp2 = input.numpy().ctypes.data_as(in2)

        #print(lib.dot_4(weight.numel(), fp1, fp2))

        M = int( weight.numel()/input.numel() )
        N = input.numel()

        out = ctypes.POINTER(ctypes.c_float)

        out = lib.mvm_4(M, N, fp1, fp2)

        result = np.ctypeslib.as_array(out, shape=(M, 1))
        result = torch.tensor(result, dtype=torch.float32).view(-1)

        if bias is not None:
            result += bias.detach()

        return torch.tensor(result, dtype=torch.float32).view(-1)


