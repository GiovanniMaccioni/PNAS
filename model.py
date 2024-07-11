import torch
import torch.nn as nn

class DepthSepConv(nn.Module):
    """
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, groups=in_channels, padding=kernel_size//2)
        self.pointwise = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

    
class DepthSepConv_block(nn.Module):
    """
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.sep1 = DepthSepConv(in_channels, in_channels, kernel_size, 1)
        self.sep2 = DepthSepConv(in_channels, out_channels, kernel_size, stride)
        self.actv = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        x = self.actv(x)
        x = self.sep1(x)
        x = self.norm1(x)
        x = self.actv(x)
        x = self.sep2(x)
        x = self.norm2(x)
        return x
    
class Conv1x7_7x1(nn.Module):
    """
    """
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size = [1, 7], padding=[0,3])
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size= [7, 1], stride=stride, padding=[3,0])
        self.actv = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.actv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.norm(x)
        return x
    
class Identity_block(nn.Module):
    def __init__(self, stride):
        super().__init__()
        
        if stride == 2:
            self.block = nn.Sequential(nn.ReLU(), nn.AvgPool2d(kernel_size=3, stride=stride ,padding=1))
        else:
            self.block = nn.Identity()
    def forward(self, x):
        x = self.block(x)

        return x
    


class Full_Convolutional_Template(nn.Module):
    """
    """
    def __init__(self, cell,in_channels, num_filters, num_blocks, num_repetitions, num_classes):
        super().__init__()

        self.seq1 = nn.Sequential(*([Cell_Template(cell, in_channels, in_channels, num_filters, num_blocks, 1, False)] +
                                    [Cell_Template(cell, num_filters*num_blocks, in_channels, num_filters, num_blocks, 1, False) for _ in range((1 if num_repetitions > 1 else 0))] +
                                    [Cell_Template(cell, num_filters*num_blocks, num_filters*num_blocks, num_filters, num_blocks, 1, False) for _ in range(num_repetitions - 2)]))
        
        self.seq2 = Cell_Template(cell, num_filters*num_blocks, (num_filters*num_blocks if num_repetitions > 1 else in_channels), num_filters*2, num_blocks, 2, False)
        num_filters = num_filters*2

        self.seq3 = nn.Sequential(*([Cell_Template(cell, num_filters*num_blocks, (num_filters//2)*num_blocks, num_filters, num_blocks, 1, True)] +
                                    [Cell_Template(cell, num_filters*num_blocks, num_filters*num_blocks, num_filters, num_blocks, 1, False) for _ in range((1 if num_repetitions > 1 else 0))] +
                                    [Cell_Template(cell, num_filters*num_blocks, num_filters*num_blocks, num_filters, num_blocks, 1, False) for _ in range(num_repetitions - 2)]))

        
        self.seq4 = Cell_Template(cell, num_filters*num_blocks, num_filters*num_blocks, num_filters*2, num_blocks, 2, False)
        num_filters = num_filters * 2

        self.seq5 = nn.Sequential(*([Cell_Template(cell, num_filters*num_blocks, (num_filters//2)*num_blocks, num_filters, num_blocks, 1, True)] +
                                    [Cell_Template(cell, num_filters*num_blocks, num_filters*num_blocks, num_filters, num_blocks, 1, False) for _ in range((1 if num_repetitions > 1 else 0))] +
                                    [Cell_Template(cell, num_filters*num_blocks, num_filters*num_blocks, num_filters, num_blocks, 1, False) for _ in range(num_repetitions - 2)]))

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(num_filters*num_blocks, num_classes)

    def forward(self, x):

        x_next = self.seq1([x,x])
        x_next = self.seq2(x_next)
        x_next = self.seq3(x_next)
        x_next = self.seq4(x_next)
        x_next = self.seq5(x_next)


        out = self.global_pool(x_next[0])
        out = out.reshape(out.shape[0], -1)
        out = self.linear(out)
        return out
    
class Cell_Template(nn.Module):
    """
    """
    def __init__(self, cell, in_channels1, in_channels2, num_filters, num_blocks, stride, prev_stride2):
        
        super().__init__()
        #Create the input list
        self.input_list = [block[0] for block in cell]

        #The stride values for the first two inputs correspond to the stride of the cells.
        #For the others, being them the output of a block within a cell, willl have stride 1, as the spatial resolution and number of channels will be already correct
        self.stride = [stride, stride]+[1]*(num_blocks + 1 - 2)

        #Create the block list for the forward
        self.block_list = nn.ModuleList([Block_Template(num_filters, num_filters, num_filters, block[1][0], block[1][1], stride1=self.stride[block[0][0]], stride2=self.stride[block[0][1]], stride=stride) for block in cell])

        #Create the conv1x1 list to corectly adjust the input spatial resolution and the number of channels if needed
        self.conv1x1_list = nn.ModuleList([nn.Identity()]*(num_blocks+1))
        #0 index means that the input comes from the previous cell so we have to only adjust the number of channels
        if (self.find_input_index(cell, 0)):
            self.conv1x1_list[0] = nn.Sequential(nn.Conv2d(in_channels1,  num_filters, 1, 1, 0), nn.ReLU())

        #1 index means that the input comes from the previous previous cell
        if (self.find_input_index(cell, 1)):
            #If the previous cell had stride 2, we have to adjust both the number of channels and the spatial resolution
            if prev_stride2:
                self.conv1x1_list[1] = nn.Sequential(nn.Conv2d(in_channels2,  num_filters, 1, 2, 0), nn.ReLU())
            #Else we only have to adjust the number of channels 
            else:
                self.conv1x1_list[1] = nn.Sequential(nn.Conv2d(in_channels2,  num_filters, 1, 1, 0), nn.ReLU())

    def forward(self, inputs):
        """
        """
        block_output_list = []

        
        for i, block in enumerate(self.block_list):
            index1 = self.input_list[i][0]
            index2 = self.input_list[i][1]

            out1 = self.conv1x1_list[index1](inputs[index1])
            out2 = self.conv1x1_list[index2](inputs[index2])

            block_output = block(out1, out2)
            block_output_list.append(block_output)
            
            #Add to the possible inputs the new block_output. Block outputs can go in input to the next block
            inputs.append(block_output)

        #Now concatenate the outputs
        cell_out = torch.cat(block_output_list, dim=1)
        
        #return to the next cell, this cell output and the previous cell output(previous in respect to this one)
        return [cell_out, inputs[0]]
    
    def find_input_index(self, cell, index):
        """
        Given a cell, find the index among the cell inputs
        """
        for block in cell:
            if (block[0][0] == index) or (block[0][1] == index):
                return True
        return False

class Block_Template(nn.Module):
    def __init__(self, in1, in2, num_filters, op1_index, op2_index, stride1, stride2, stride):
        super().__init__()
        
        self.op1 = self.select_operation(op1_index, in1, num_filters, stride1)
        self.op2 = self.select_operation(op2_index, in2, num_filters, stride2)
        

    def forward(self, x1, x2):

        x1 = self.op1(x1)
        x2 = self.op2(x2)

        x_sum = x1 + x2
        return x_sum
    
    def select_operation(self, op_index, in_channels = None, out_channels = None, stride = None):
        if op_index == 0:
            operation = DepthSepConv_block(in_channels, out_channels, kernel_size=3, stride=stride)
        elif op_index == 1:
            operation = DepthSepConv_block(in_channels, out_channels, kernel_size=5, stride=stride)
        elif op_index == 2:
            operation = DepthSepConv_block(in_channels, out_channels, kernel_size=7, stride=stride)
        elif op_index == 3:
            operation = Conv1x7_7x1(in_channels, out_channels, stride)
        elif op_index == 4:
            operation = Identity_block(stride=stride)
        elif op_index == 5:
            operation = nn.AvgPool2d(kernel_size=3, stride=stride ,padding=1)
        elif op_index == 6:
            operation = nn.MaxPool2d(kernel_size=3, stride=stride ,padding=1)
        elif op_index == 7:
            operation = nn.Sequential(nn.ReLU(), nn.Conv2d(in_channels, out_channels, 3, stride=stride, dilation=2, padding=2))
        else:
            print("Not valid operation")
    
        return operation