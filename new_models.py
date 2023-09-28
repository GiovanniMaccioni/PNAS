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
    def __init__(self, input1, input2, op1_index, op2_index, stride, num_filters):
        super().__init__()
        self.op1_index = op1_index
        self.op2_index = op2_index

        self.op1, self.op2, self.out_dim, self.out_ch1, self.out_ch2 = self.create_operations(input1, input2, stride, num_filters)
        #return input dimension and other info


    #FIXME i put first_cell and number_of_filters in the forward call, but i want to change that
    def forward(self, input1, input2):

        o1 = self.op1(input1)    
        o2 = self.op2(input2)

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
        
        op1 = select_operation(self.op1_index, input1.shape[1], out_channels=out_channels1, stride=stride1)
        op2 = select_operation(self.op2_index, input2.shape[1], out_channels=out_channels2, stride=stride2)

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
        out_dim =  out_dim2_postop2
        out_ch1 = out_channels1
        out_ch2 = out_channels2
        
        if out_dim1_postop1 > out_dim2_postop2:
            conv1d = nn.Conv2d(out_channels1, out_channels2, 1, stride=out_dim1_postop1//out_dim2_postop2).to(device)
            #TODO make a sequential with op1
            op1 = nn.Sequential(op1, conv1d)#TOCHECK Maybe I have to remove the input channels wit the sequential
            out_dim =  out_dim2_postop2
            out_ch1 = out_channels2

        elif out_dim1_postop1 < out_dim2_postop2:
            conv1d = nn.Conv2d(out_channels2,  out_channels1, 1, stride=out_dim2_postop2//out_dim1_postop1).to(device)
            #TODO make a sequential with op2
            op2 = nn.Sequential(op2, conv1d)
            out_dim =  out_dim1_postop1
            out_ch2 = out_channels1

        elif out_channels1 > out_channels2:
            conv1d = nn.Conv2d(out_channels2,  out_channels1, 1, stride=1).to(device)
            #TODO make a sequential with op2
            op2 = nn.Sequential(op2, conv1d)
            out_ch2 = out_channels1

        elif out_channels1 < out_channels2:
            conv1d = nn.Conv2d(out_channels1,  out_channels2, 1, stride=1).to(device)
            #TODO make a sequential with op1
            op1 = nn.Sequential(op1, conv1d)
            out_ch1 = out_channels2


        return op1, op2, out_dim, out_ch1, out_ch2
        #return max(out_channels1, out_channels2), min(input1.shape[2]//stride, input2.shape[2]//stride)

    def get_next_inputs(self):
        return torch.randn((1, self.out_ch1, self.out_dim, self.out_dim)), torch.randn((1, self.out_ch2, self.out_dim, self.out_dim))
 

class Cell_Template(nn.Module):
    """
    Idea: Pass the cell as an object, and define the model with its forward here

    """
    #TOCHECK probably i have to put the filters of the 1D Convolutions as parameters of the cell!!!!
    def __init__(self, stride, num_filters, cell, input1, input2):
        super().__init__()

        """self.stride = stride
        self.num_filters = num_filters"""

        self.blocks_list, self.inputs_list, self.conv1d_list, self.next_input = self.build_cell(cell,  stride, num_filters, input1, input2)

    def forward(self, inputs_):
        """
        inputs_: list; at first it will contain two copies of the input images. As the cell is deeper in the architecture,
                    this list will contain the outputs of the previous cells and/or outputs from previous blocks(if the 
                    cell contains more than one block).
        """
        block_output_list = []#TODO Substitute with torch.cat

        #This cycle will populate the blocks, and give us the block outputs. We 
        #save the min dimension to eventually apply a 1D Convolution and have all the 
        #block_outputs all of the same Height and Width before concatenation

        for i, block in enumerate(self.blocks_list):
            index1 = self.inputs_list[i][0]
            index2 = self.inputs_list[i][1]

            block_output = block(inputs_[index1], inputs_[index2])
            block_output_list.append(block_output)
            
            #Add to the possible inputs, th new block_output. It can go in input to the next block
            inputs_.append(block_output)

        for i in range(len(block_output_list)):
            if self.conv1d_list[i] != None:
                block_output_list[i] = self.conv1d_list[i](block_output_list[i])

        #Now concatenate the outputs
        cell_out = torch.cat(block_output_list, dim=1)

        #out = torch.cat((cell_out, inputs_[:, 0][:, None]))
        out = [cell_out, inputs_[0]]
        return out
    
    def build_cell(self, cell, stride, num_filters, input1, input2):#TOCHECK maybe I don't need num_filters=None
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

        in_id_list = []
        op_id_list = []

        inputs_list = [input1, input2]
        block_list = nn.ModuleList()

        #TODO if here I create a list of inputs, maybe I can fix the forward method in BlockTemplate
        for block in cell:
            in_id_list.append([block[0][0], block[0][1]])
            op_id_list.append([block[1][0], block[1][1]])

            #TOCHECK I don't know if the addressing is correct
            block_ = Block_Template(inputs_list[in_id_list[-1][0]], inputs_list[in_id_list[-1][1]], op_id_list[0][0], op_id_list[0][1], stride, num_filters)
            input1, input2 = block_.get_next_inputs()
            block_list.append(block_)
            #TOCHECK As i Have to append the sum of the otput of the block, the dimension
            #should be the same as in the block_ we have the 1d Convolutions!
            inputs_list.append(input1)
        
        block_output_list = inputs_list[2:]
        #TOCHECK Like this I take the minimum dimension for concatenation
        min_dim = block_output_list[0].shape[2]
        min_dim = [b.shape[2] for b in block_output_list if min_dim >= b.shape[2]]
        min_dim = min_dim[-1]

        conv1d_list = nn.ModuleList()
        for i in range(len(block_output_list)):
            #
            if block_output_list[i].shape[2] > min_dim:
                conv1d_list.append(nn.Conv2d(block_output_list[i].shape[1], block_output_list[i].shape[1], 1, stride = block_output_list[i].shape[2]//min_dim))
                #Simulate the effect of the 1d Convolution
                block_output_list[i] = torch.randn(1, block_output_list[i].shape[1], min_dim, min_dim)
            else:
                conv1d_list.append(None)

        #Now concatenate the outputs
        next_input = torch.cat(block_output_list, dim=1)

        #out = torch.cat((cell_out, inputs_[:, 0][:, None]))
        #Input for the next cell!!!

        return block_list, in_id_list, conv1d_list, next_input
    
    def get_next_input(self):
        return self.next_input

class Full_Convolutional_Template(nn.Module):
    """
    Idea: Pass the cell as an object, and define the model with its forward here
    """
    def __init__(self, cell, num_repetitions, num_filters, num_classes, input1, input2):
        super().__init__()

        #TODO from the previous cell I have to know the new inputs for the next cell!!
        self.seq1, input_list = self.build_(num_repetitions, input1, input2, num_filters, cell, 1)
    
        self.seq2 = Cell_Template(2, num_filters, cell, input_list[0], input_list[1])
        new_input = self.seq2.get_next_input()
        input_list = [new_input, input_list[0]]

        self.seq3, input_list = self.build_(num_repetitions, input_list[0], input_list[1], num_filters, cell, 1)

        self.seq4 = Cell_Template(2, num_filters, cell, input_list[0], input_list[1])
        new_input = self.seq4.get_next_input()
        input_list = [new_input, input_list[0]]

        self.seq5, input_list = self.build_(num_repetitions, input_list[0], input_list[1], num_filters, cell, 1)

        #I have to extract from the last cell output,  that is  a list, the last output
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        #TOCHECK Foir now I use the las input information to fix the linear layer input_channels
        self.linear = nn.Linear(input_list[0].shape[1], num_classes)#TOCHECK It says softmax classification layer

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
    
    def build_(self, num_repetitions, input1, input2, num_filters, cell, stride):
        input_list = [input1, input2]
        cell_list = nn.ModuleList()
        for i in range(num_repetitions):
            cell_ = Cell_Template(stride, num_filters, cell, input_list[0], input_list[1])
            new_input = cell_.get_next_input()
            input_list = [new_input, input_list[0]]
            cell_list.append(cell_)
        
        seq = nn.Sequential(*cell_list)
        return seq, input_list 


    
