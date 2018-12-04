import ctypes

lib = ctypes.cdll.LoadLibrary('./exe/exe.so')

import torch
from torch.autograd import Function
from torch.nn import Module
import torch.nn as nn

# ------------------------------------------------------------------------------ Clover Dot_4 Network
class Dot4Module(Module):
    def forward(self, input1, input2):
        return Dot4Function()(input1, input2)

class Dot4Function(Function):
    def forward(self, input1, input2):
        lib.dot_4.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        lib.dot_4.restype  = ctypes.c_float

        in1 = ctypes.POINTER(ctypes.c_float)
        in2 = ctypes.POINTER(ctypes.c_float)

        fp1 = input1.numpy().ctypes.data_as(in1)
        fp2 = input2.numpy().ctypes.data_as(in2)

        #print(lib.dot_4(input1.numel(), fp1, fp2))

        return torch.tensor(lib.dot_4(input1.numel(), fp1, fp2), dtype=torch.float32)

class Dot4Network(nn.Module):
    def __init__(self):
        super(Dot4Network, self).__init__()
        self.exe = Dot4Module()

    def forward(self, input1, input2):
        return self.exe(input1, input2)

# ------------------------------------------------------------------------------ torch Dot_32 Network
class TorModule(Module):
    def forward(self, input1, input2):
        return TorFunction()(input1, input2)

class TorFunction(Function):
    def forward(self, input1, input2):
        return torch.dot(input1, input2)

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

#file = open("time_dot_TT.csv", 'a')
#file.write("i, clover-4-bit dot, clover-8-bit dot, torch-fp32-bit dot \n")
#file = open("loss_dot_uniform.csv", 'a')
#file.write("i, loss(4), loss(8) \n")

N = 200

import torchvision.models as md
dn = md.densenet121(pretrained=True)

st = dn.state_dict()
input1 = st['features.denseblock4.denselayer15.conv1.weight'].view(-1)
input2 = st['features.denseblock4.denselayer15.conv1.weight'].view(-1)
i = input1.numel()
print(i)

for key, value in st:
 if '.weight' in key:
  sum_c8 = 0
  sum_c4 = 0
  n = 1
  while n <= N:
    #input1 = torch.rand(i, dtype=torch.float32)
    #input2 = torch.rand(i, dtype=torch.float32)
    sta_to = time.time()
    res_to = model_to(input1, input2)
    end_to = time.time()

    sta_c4 = time.time()
    res_c4 = model_c4(input1, input2)
    end_c4 = time.time()

    """
    print(i)
    print( "clover time(4) : %.9f " % (end_c4 - sta_c4) )
    print( "normal time(32): %.9f " % (end_to - sta_to) )
    #file.write(str(i) + ", " + str(end_c4 - sta_c4) + ", " + str(end_c8 - sta_c8) + ", " + str(end_to - sta_to) + " \n")
    """

    sum_c4 += 100 * ((res_c4 - res_to)/res_to).abs()

    n += 1

  #file.write(str(i) + ", " + str((sum_c4/N).numpy()) + ", " + str((sum_c8/N).numpy()) + " \n")
  print("Vector size: ", i)
  print( "AVG.(%d) c4 loss: %.5f %% " % (N, sum_c4/N) )


#file.close()

"""
file = open("dis.csv", 'w')
input1 = torch.rand(512)
for i in range(0, 512):
  file.write(str(input1.numpy()[i]) + ", \n")
file.close()
"""
