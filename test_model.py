import torch

import models as m
import torch.nn as nn

import itertools

import string
import compose as comp

B = 5 #Maximum number of blocks
K = 256#Number of Model samples
N = 6#Number of repetition of the cells with stride 1<-------"CHECK HERE"!!!
F = 32#Number of filters in the first layer
#TOCHECK N = 2, F = 4 ---> group=in_channels out_channels must divide groups

#[ H_prev_cell, H_(prev_cell-1), H_1, ..., H_(B-1)]
inputs_indices = torch.arange(0,2+B-1)#2 pevious cell, B-1 as the last block can also have B-1 possible inputs from previous blocks in the same cell

#[3x3_DWC, 5x5_DWC, 7x7_DWC, 1x7_7x1_Conv, Id, 3x3_AvgPool, 3x3_MaxPool, 3x3_DilatedConv]
operations_indices = torch.arange(0,8)#number of operations considered is 8

#Have to transform them in tensors in order to use one hot encoding!

one_hot_inputs = torch.nn.functional.one_hot(inputs_indices)
one_hot_operations = torch.nn.functional.one_hot(operations_indices)

one_hot_inputs = one_hot_inputs.tolist()
one_hot_operations = one_hot_operations.tolist()

x = comp.get_combinations(one_hot_inputs[:2], one_hot_operations)
#x = flatten2(x)
#----> da allenare
tr = [[elem] for elem in x]
#x = [x]
model_list_1 = []
for cell in tr:
    model_list_1.append(m.Full_Convolutional_Template(cell, N, F, 10))

y = comp.get_combinations(one_hot_inputs[:3], one_hot_operations)
#y = flatten(y)

tr = list(itertools.product(x, y))
tr = tr[1000:1000+K]

model_list_2 = []
for cell in tr:
    model_list_2.append(m.Full_Convolutional_Template(cell, N, F, 10))

#------test whole model

model = model_list_2[0]
#print(model)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)


x = [torch.randn(8, 3, 32, 32, requires_grad=True).to(device), torch.randn(8, 3, 32, 32, requires_grad=True).to(device)]
#x = torch.randn((64, 3, 32, 32), requires_grad=True)
#x = x.to(device)
#with torch.cuda.amp.autocast():
torch_out = model(x)

print(torch_out.shape)

print(model)

nparameters = sum(p.numel() for p in model.parameters())
print("Number of Parameters: ", nparameters)