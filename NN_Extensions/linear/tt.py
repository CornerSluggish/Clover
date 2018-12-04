import ctypes

lib = ctypes.cdll.LoadLibrary('./exe/exe.so')

import torch
from torch.autograd import Function
from torch.nn import Module
import torch.nn as nn

# ------------------------------------------------------------------------------ Clover Dot_4 Network
def DD4(input1, input2):
        lib.dot_4.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        lib.dot_4.restype  = ctypes.c_float

        in1 = ctypes.POINTER(ctypes.c_float)
        in2 = ctypes.POINTER(ctypes.c_float)

        fp1 = input1.numpy().ctypes.data_as(in1)
        fp2 = input2.numpy().ctypes.data_as(in2)

        return torch.tensor(lib.dot_4(input1.numel(), fp1, fp2), dtype=torch.float32)

# ------------------------------------------------------------------------------ Clover Dot_8 Network
def DD8(input1, input2):
        lib.dot_8.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        lib.dot_8.restype  = ctypes.c_float

        in1 = ctypes.POINTER(ctypes.c_float)
        in2 = ctypes.POINTER(ctypes.c_float)

        fp1 = input1.numpy().ctypes.data_as(in1)
        fp2 = input2.numpy().ctypes.data_as(in2)

        #print(lib.dot_8(input1.numel(), fp1, fp2))

        return torch.tensor(lib.dot_8(input1.numel(), fp1, fp2), dtype=torch.float32)

class Dot8Network(nn.Module):
    def __init__(self):
        super(Dot8Network, self).__init__()
        self.exe = Dot8Module()

    def forward(self, input1, input2):
        return self.exe(input1, input2)

# ------------------------------------------------------------------------------ Test Multiple-size Input Vector




import time

#file = open("time_dot_TT.csv", 'a')
#file.write("i, clover-4-bit dot, clover-8-bit dot, torch-fp32-bit dot \n")


#import sys
#i = int(sys.argv[1])

N = 2
i = 8192
sum_to = 0.
sum_c8 = 0.
sum_c4 = 0.
n = 1
while n <= N:
#if i <= 33554432:
    input1 = torch.randn(i, dtype=torch.float32)
    input2 = torch.randn(i, dtype=torch.float32)
    sta_to = time.time()
    res_to = torch.dot(input1, input2)
    end_to = time.time()

    #input1 = torch.randn(i, dtype=torch.float32)
    #input2 = torch.randn(i, dtype=torch.float32)
    sta_c8 = time.time()
    res_c8 = DD8(input1, input2)
    end_c8 = time.time()

    #input1 = torch.randn(i, dtype=torch.float32)
    #input2 = torch.randn(i, dtype=torch.float32)
    sta_c4 = time.time()
    res_c4 = DD4(input1, input2)
    end_c4 = time.time()


    #print(i)
    #print( "clover time(4) : %.9f " % (end_c4 - sta_c4) )
    #print( "clover time(8) : %.9f " % (end_c8 - sta_c8) )
    #print( "normal time(32): %.9f " % (end_to - sta_to) )
    #print("\n")

    sum_to += end_to - sta_to
    sum_c8 += end_c8 - sta_c8
    sum_c4 += end_c4 - sta_c4

    #file.write(str(i) + ", " + str(end_c4 - sta_c4) + ", " + str(end_c8 - sta_c8) + ", " + str(end_to - sta_to) + " \n")

    n += 1

#file.close()

print(i)
print( "AVG. clover time(4) : %.9f " % (sum_c4/N) )
print( "AVG. clover time(8) : %.9f " % (sum_c8/N) )
print( "AVG. normal time(32): %.9f " % (sum_to/N) )

