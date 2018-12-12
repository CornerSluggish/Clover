import ctypes

lib = ctypes.cdll.LoadLibrary('./fully_con/fully_con.so')

import torch
from torch.autograd import Function
from torch.nn import Module
import torch.nn as nn



import numpy as np




# ------------------------------------------------------------------------------ Clover Dot_4 Network
class Linear4b(Module):
    def forward(self, input1, input2):
        return Mvm4Function()(input1, input2)

class Mvm4Function(Function):
    def forward(self, input1, input2):
        lib.mvm_4.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        lib.mvm_4.restype  = ctypes.POINTER(ctypes.c_float)

        in1 = ctypes.POINTER(ctypes.c_float)
        in2 = ctypes.POINTER(ctypes.c_float)

        fp1 = input1.numpy().ctypes.data_as(in1)
        fp2 = input2.numpy().ctypes.data_as(in2)

        #print(lib.dot_4(input1.numel(), fp1, fp2))

        M = int( input1.numel()/input2.numel() )
        N = input2.numel()

        out = ctypes.POINTER(ctypes.c_float)

        out = lib.mvm_4(M, N, fp1, fp2)

        result = np.ctypeslib.as_array(out, shape=(M, 1))

        return torch.tensor(result, dtype=torch.float32).view(-1)

class Dot4Network(nn.Module):
    def __init__(self):
        super(Dot4Network, self).__init__()
        self.exe = Linear4b()

    def forward(self, input1, input2):
        return self.exe(input1, input2)

# ------------------------------------------------------------------------------ torch Dot_32 Network
class TorModule(Module):
    def forward(self, input1, input2):
        return TorFunction()(input1, input2)

class TorFunction(Function):
    def forward(self, input1, input2):
        return torch.mv(input1, input2)

class TorNetwork(nn.Module):
    def __init__(self):
        super(TorNetwork, self).__init__()
        self.tor = TorModule()

    def forward(self, input1, input2):
        return self.tor(input1, input2)

# ------------------------------------------------------------------------------ Test Multiple-size Input Vector





model_c4 = Dot4Network()
model_to = TorNetwork()

import time

M = 4096
N = 512

input1 = torch.rand(M, N, dtype=torch.float32)
input2 = torch.rand(N, dtype=torch.float32)

#print(input1, input2)

sta_to = time.time()
res_to = model_to(input1, input2)
end_to = time.time()

sta_c4 = time.time()
res_c4 = model_c4(input1, input2)
end_c4 = time.time()


#print(res_to)
#print(res_c4)

#print( np.unique(res_c4.numpy()) )


#"""
print(M, N)
print( "clover time(4) : %.9f " % (end_c4 - sta_c4) )
print( "normal time(32): %.9f " % (end_to - sta_to) )
#"""

#print("Vector size: ", i)


