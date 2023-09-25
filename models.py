import torch
import torch.nn as nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class DepthSepConv(nn.Module):#TODO change name
    """
    Valid convolutions; only the stride reduces the height and width!!
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        #I put stride in the forst convolution as the 1d convolution would totally skip some pixels with a stride
        #greater than 1
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, groups=in_channels, padding=kernel_size//2)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

    
class DepthSepConv_block(nn.Module):#TODO change name
    """
    
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.sep1 = DepthSepConv(in_channels, in_channels, kernel_size, 1)
        self.sep2 = DepthSepConv(in_channels, out_channels, kernel_size, stride)
        self.actv = nn.ReLU()
        #self.norm = nn.BatchNorm2d()FIXME missing num_features!!

        #self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1)


    def forward(self, x):
        x = self.actv(x)
        x = self.sep1(x)
        #x = self.norm(x)
        x = self.actv(x)
        x = self.sep2(x)
        #x = self.norm(x)
        return x
    
class Block2(nn.Module):#FIXME I don't know how to deal with number of channels
    """
    two subsequent convolutions 1x7, 7x1
    """
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size = [1, 7], padding=[0,3])
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size= [7, 1], stride=stride, padding=[3,0])
        self.actv = nn.ReLU()#TOCHECK I don't know if there is a RelU here!!

    def forward(self, x):
        x = self.conv1(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x
    

#TOCHECK i don't know if the rest of the blocks are to be made classes

#Identity

#3x3 average pooling

#3x3 max pooling

#3x3 Dilated Convolution

#1x1 Convolution to adjust dimension if needed

def select_operation(op_index, in_channels = None, out_channels = None, stride = None):
    if op_index == 0:
        operation = DepthSepConv_block(in_channels, out_channels, kernel_size=3, stride=stride)
    elif op_index == 1:
        operation = DepthSepConv_block(in_channels, out_channels, kernel_size=5, stride=stride)
    elif op_index == 2:
        operation = DepthSepConv_block(in_channels, out_channels, kernel_size=7, stride=stride)
    elif op_index == 3:
        operation = Block2(in_channels, out_channels, stride)
    elif op_index == 4:
        operation = nn.Identity()
    elif op_index == 5:
        operation = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)#TOCHECK stride??
    elif op_index == 6:
        operation = nn.MaxPool2d(kernel_size=3, stride=stride,padding=1)#TOCHECK stride??
    elif op_index == 7:
        operation = nn.Conv2d(in_channels, out_channels, 3, stride=stride, dilation=2, padding=2)#TOCHECK dilation value is not specified anywhere
    else:
        print("Not valid operation")#TODO replace with error handling!
    
    return operation.to(device)

class Block_Template(nn.Module):
    """
    Idea: Pass the cell as an object, and define the model with its forward here
    """
    def __init__(self, op1_index, op2_index):
        super().__init__()
        self.op1_index = op1_index
        self.op2_index = op2_index
        self.add_modules = True

    #FIXME i put first_cell and number_of_filters in the forward call, but i wat to channge that
    def forward(self, input1, input2, stride, num_filters):

        if self.add_modules:
            self.create_operations(input1, input2, stride, num_filters)

        o1 = self.op1(input1)    
        o2 = self.op2(input2)

        if o1.shape[2] > o2.shape[2]:
            o1 = self.conv1d(o1)

        elif o1.shape[2] < o2.shape[2]:
            o2 = self.conv1d(o2)
        
        elif o1.shape[1] > o2.shape[1]:
            o2 = self.conv1d(o2)

        elif o1.shape[1] < o2.shape[1]:
            o1 = self.conv1d(o1)

        out_block = o1 + o2
        
        return out_block
    
    def create_operations(self, input1, input2, stride, num_filters=None):
        """
        This method is used to initialize the convolutional operations at run time, as the in_channels dimension
        is determined by the input, chosen at run time.
        """
        
        if num_filters != None:
            out_channels1, out_channels2 = num_filters, num_filters
        else:
            out_channels1 = input1.shape[1]*stride
            out_channels2 = input2.shape[1]*stride
        
        if self.op1_index in [4,5,6]:
            out_channels1 = input1.shape[1]
        
        if self.op2_index in [4,5,6]:
            out_channels2 = input2.shape[1]

        if self.op1_index==4:
            stride1 = 1
        else:
            stride1=stride
        
        if self.op2_index==4:
            stride2 = 1
        else:
            stride2 = stride
        
        self.op1 = select_operation(self.op1_index, input1.shape[1], out_channels=out_channels1, stride=stride1)
        self.op2 = select_operation(self.op2_index, input2.shape[1], out_channels=out_channels2, stride=stride2)

        #Here I may need to perform a 1D convolution to have the same Height, Width and number of channels to perform the sum
        #of the putputs of the operations
        
        out_dim1_postop1 = input1.shape[2]//stride1
        out_dim2_postop2 = input2.shape[2]//stride2

        """if out_channels1 > out_channels2:
            out_channels = out_channels1#TOCHECK I think i hav to take the bigger one
        #elif out_channels1 < out_channels2:
        else:
            out_channels = out_channels2
        
        out_dim1_postop1//out_dim2_postop2"""
        
        if out_dim1_postop1 > out_dim2_postop2:
            self.conv1d = nn.Conv2d(out_channels1, out_channels2, 1, stride=out_dim1_postop1//out_dim2_postop2).to(device)

        elif out_dim1_postop1 < out_dim2_postop2:
            self.conv1d = nn.Conv2d(out_channels2,  out_channels1, 1, stride=out_dim2_postop2//out_dim1_postop1).to(device)
        
        elif out_channels1 > out_channels2:
            self.conv1d = nn.Conv2d(out_channels2,  out_channels1, 1, stride=1).to(device)
        
        elif out_channels1 < out_channels2:
            self.conv1d = nn.Conv2d(out_channels1,  out_channels2, 1, stride=1).to(device)

        self.add_modules = False
        
        #return max(out_channels1, out_channels2), min(input1.shape[2]//stride, input2.shape[2]//stride)

def build_cell2(cell,  stride, num_filters=None):
    """
    After finding the top K accuracies cells,
    we will transform them in cell modules, pass them to the build_model function to obtain the final convolution

    For now we will transform a cell at a time

    cell: sequence of inputs and outputs; [I1, I2, O1, O2]*number_of_blocks<--------NO!!!!!!!!
    operations: sequence of coupled operations: (O1, O2)*num_blocks
    """
    #TOCHECK I don't know if the cell will come as separate sequences, (I1, I2)*num_blocks (O1, O2)*num_blocks or not
    #For now I suppose a cell is [I1, I2, O1, O2]*number_of_blocks----> NO!!!!
    #Maybe I need only the operations<-------- For now I go with this

    """for i in range(0, len(operations), 2):
        op1 = select_operation(operations[i])
        op2 = select_operation(operations[i+1])"""
    
    """
    - Create an input list
    - Create an operation list(maybe optional)
    - 
    """
    inputs_list = []
    #operations_list = []
    block_list = nn.ModuleList()
    for block in cell:
        inputs_list.append([block[0][0], block[0][1]])
        #operations_list.append(block[1])
        block_list.append(Block_Template(block[1][0], block[1][1]))
    cell_ = Cell_Template(inputs_list, block_list, stride, num_filters).to(device)

    return cell_   

class Cell_Template(nn.Module):
    """
    Idea: Pass the cell as an object, and define the model with its forward here

    """
    #TOCHECK probably i have to put the filters of the 1D Convolutions as parameters of the cell!!!!
    def __init__(self, inputs_list, block_list, stride, num_filters):
        super().__init__()
        self.inputs_list = inputs_list
        self.blocks_list = block_list

        self.add_modules = True
        self.conv1d_list = nn.ModuleList()
        #--------
        self.stride = stride
        self.num_filters = num_filters

    def forward(self, inputs_):
        block_output_list = []#TODO Substitute with torch.cat
        max_channels = 0

        #height and width since we work with square images
        min_dim = 10000#TOCHECK initialization

        #With this cycle i create the operations inside the blocks, and i obtain the channels output to initialize 
        #the 1d convolutions to permorm the concats
        """for i, block in enumerate(self.blocks_list):
            index1 = self.inputs_list[i][0]
            index2 = self.inputs_list[i][1]

            tmp_max_out, tmp_min_dim = block.create_operations(inputs_[index1], inputs_[index2], self.stride, self.num_filters)

            if tmp_max_out > max_channels:
                max_channels = tmp_max_out

            if tmp_min_dim <  min_dim:
                min_dim = tmp_min_dim"""
        #This cycle will populate the blocks, and give us the block outputs. We 
        #save the min dimension to eventually apply a 1D Convolution and have all the 
        #blockoutputs all of the same Height and Width before concatenation
        min_dim = 1000
        for i, block in enumerate(self.blocks_list):
            index1 = self.inputs_list[i][0]
            index2 = self.inputs_list[i][1]

            block_output = block(inputs_[index1], inputs_[index2], self.stride, self.num_filters)
            block_output_list.append(block_output)
            if block_output.shape[2] <  min_dim:
                min_dim = block_output.shape[2]
            
            #Add to the possible inputs, th new blockoutput. It can go in input to the next block
            inputs_.append(block_output)

            #inputs_.append(block_output)
        #In this Cycle we add the 1D convolution for the final concatenation. 
        for i in range(len(block_output_list)):
            index_conv1d = 0
            #
            if block_output_list[i].shape[2] > min_dim:
                if self.add_modules:
                    self.conv1d_list.append(nn.Conv2d(block_output_list[i].shape[1], block_output_list[i].shape[1], 1, stride = block_output_list[i].shape[2]//min_dim).to(device))#TOCHECK I don't know what padding is convinient here
                block_output_list[i] = self.conv1d_list[index_conv1d](block_output_list[i])
                index_conv1d = index_conv1d + 1
        
        self.add_modules = False

        #Now concatenate the outputs
        cell_out = torch.cat(block_output_list, dim=1)

        #out = torch.cat((cell_out, inputs_[:, 0][:, None]))
        out = [cell_out, inputs_[0]]
        return out


class Full_Convolutional_Template(nn.Module):
    """
    Idea: Pass the cell as an object, and define the model with its forward here
    """
    def __init__(self, cell, num_repetitions, num_filters, num_classes):
        super().__init__()
        #self.seq1 = nn.Sequential(*([build_cell2(cell, 1, num_filters=num_filters)]\
        #                        + [build_cell2(cell, 1) for _ in range(num_repetitions-1)]))
        self.seq1 = nn.Sequential(*([build_cell2(cell, 1, num_filters=num_filters) for _ in range(num_repetitions)]))
    
        self.seq2 = build_cell2(cell, 2, num_filters=num_filters)

        #self.seq3 = nn.Sequential(*([build_cell2(cell, 1) for _ in range(num_repetitions)]))
        #self.seq3 = nn.Sequential(*([build_cell2(cell, 1, num_filters=num_filters)]\
        #                        + [build_cell2(cell, 1) for _ in range(num_repetitions-1)]))
        self.seq3 = nn.Sequential(*([build_cell2(cell, 1, num_filters=num_filters) for _ in range(num_repetitions)]))


        self.seq4 = build_cell2(cell, 2, num_filters=num_filters)

        #self.seq5 = nn.Sequential(*([build_cell2(cell, 1) for _ in range(num_repetitions)]))
        #self.seq5 = nn.Sequential(*([build_cell2(cell, 1, num_filters=num_filters)]\
        #                        + [build_cell2(cell, 1) for _ in range(num_repetitions-1)]))
        self.seq5 = nn.Sequential(*([build_cell2(cell, 1, num_filters=num_filters) for _ in range(num_repetitions)]))



        #I have to extract from the last cell output,  that is  a list, the last output
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.LazyLinear(num_classes)#TOCHECK It says softmax classification layer

    def forward(self, x):
        """
        x: Image of the dataset
        """
        x = self.seq1(x)
        x = self.seq2(x)
        x = self.seq3(x)
        x = self.seq4(x)
        x = self.seq5(x)
        
        x = self.global_pool(x[0])
        x = x.reshape(x.shape[0], -1)#TOCHECK
        x = self.linear(x)
        return x
    
