import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import MNIST
from torch.utils.data import Subset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from itertools import chain

class ResNet_Block(nn.Module):
    """
        in_channels: Number of channels of the input image
        out_channels: Number of output channels of the convolution block
        kernel_size:  Dimension of the kernel 
        stride: stride
        padding: padding     
    """
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, padding = padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.actv = nn.ReLU()

    def forward(self, x, proj = None):
        x = self.conv(x)
        x = self.norm(x)
        if proj != None:
            #proj is the output of a residual connection
            x = x + proj
        x = self.actv(x)
        return x

    
class ResNet_Layer(nn.Module):
    """
        in_channels: Number of channels of the input feature map
        out_channels: Number of output channels of the convolution block
        num_conv:  Number of convolutional layers to stack 
        residual_step: Number of convolution layers to skip
        first: Indicates if it is the first ResNet Layer; it needs a different initialization
    """
    def __init__(self, in_channels, out_channels, num_conv, residual_step = 0, first = False):
        super().__init__()
        
        if first == True:
            stride_first = 1
            in_channels_first = in_channels
        else:
            stride_first = 2
            in_channels_first = in_channels // 2
        
        self.block = nn.ModuleList([ResNet_Block(in_channels_first, out_channels, 3, stride_first, padding = 1)]\
                                + [ResNet_Block(in_channels, out_channels, 3, stride = 1, padding = 1) for _ in range(num_conv-1)])
        self.residual_step = residual_step

        if residual_step != 0:
            assert residual_step <= num_conv , "residual_step can't be above num_conv"
            assert num_conv % residual_step == 0 , "residual_step should be a dividor for num_conv"
            self.num_shortcuts = num_conv // residual_step
            if first == True:
                self.projections = nn.ModuleList([nn.Identity() for _ in range(self.num_shortcuts)])
            else:
                self.projections = nn.ModuleList([nn.Conv2d(out_channels//2, out_channels, 1, 2)]\
                                                 + [nn.Identity() for _ in range(self.num_shortcuts - 1)])


    def forward(self,x):

        if self.residual_step != 0:
            count = self.num_shortcuts
            for i in range(0, len(self.block), self.residual_step):
                print("---FOR---")
                print("i:", i)
                print("self.residual_step:",  self.residual_step)
                print("self.num_shortcuts:",  self.num_shortcuts)
                print("self.projections[-count]:",  self.projections[-count])
                print("len(self.block):",  len(self.block))
                proj = self.projections[-count](x)
                for j in range(i, i + self.residual_step):
                    print("---FOR---")
                    print("j: ", j)
                    if j != (i + self.residual_step - 1):
                        x = self.block[j](x)
                    else:
                        print("---ELSE---")
                        print("j: ", j)
                        x = self.block[j](x, proj)
                #x = x + proj
                count -= 1
        else:
            for i in range(len(self.block)):
                x = self.block[i](x)
        
        return x
    

class ResNet(nn.Module):
    """
        in_channels: Number of channels of the input image
        out_channels: Number of output channels of the first convolution block
        num_classes: Number of classes to classify
        num_conv_per_level: Number of convolution in The ResNet Block
        num_levels: Number of ResNet Blocks
        residual_step: How many Convolutions are skipped
    """
    def __init__(self, in_channels, out_channels, num_classes, num_conv_per_level, num_levels, residual_step):
        super().__init__()
        self.conv_in = ResNet_Block(in_channels, out_channels, 7, 2, 3)
        self.pool = nn.MaxPool2d(3, 2)

        assert num_conv_per_level > 0, "0 not allowed in num_conv_per_level; at least 1"
        
        self.res_block_list = nn.Sequential(*([ResNet_Layer(out_channels, out_channels, num_conv_per_level, residual_step, first = True)]\
                                            +[ResNet_Layer(out_channels*2**i, out_channels*2**i, num_conv_per_level, residual_step) for i in range(1, num_levels)]))

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(out_channels*2**(num_levels-1), num_classes)

    def forward(self,x):
        x = self.conv_in(x)
        x = self.pool(x)

        """for res_block in self.res_block_list:
            x = res_block(x)"""
        x = self.res_block_list(x)

        x = self.global_pool(x)
        x = x.reshape(x.shape[0], -1)#TOCHECK
        x = self.linear(x)

        return x



model = ResNet(3, 64, 10, 8, 1, 0)
print(model)


x = torch.randn(64, 3, 32, 32, requires_grad=True)
torch_out = model(x)

# Export the model
torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "./Resnet_nl1_nc8.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})