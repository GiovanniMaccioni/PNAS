import torch

import models as m
import torch.nn as nn

import itertools

import string

#combinations = list(itertools.product(strategy_list, point_list, roll_list, prob_list))

B = 5 #Maximum number of blocks
K = 256#Number of Model samples
N = 10#Number of repetition of the cells with stride 1
F = 24#Number of filters in the forst layer

accuracies = []

#[H_1, ..., H_(B-1), H_prev_cell, H_(prev_cell-1)]
inputs_indices = list(range(2+B-1))#2 previous cell, B-1 as the last block can also have B-1 possible inputs from previous blocks in the same cell
inputs_indices = list(string.ascii_uppercase)[:2+B-1]

#[3x3_DWC, 5x5_DWC, 7x7_DWC, 1x7_7x1_Conv, Id, 3x3_AvgPool, 3x3_MaxPool, 3x3_DilatedConv]
operations_indices = list(range(8))#number of operations considered is 8

#----------------WITH TENSORS--------------------

#[H_1, ..., H_(B-1), H_prev_cell, H_(prev_cell-1)]
inputs_indices = torch.arange(0,2+B-1)#2 pevious cell, B-1 as the last block can also have B-1 possible inputs from previous blocks in the same cell

#[3x3_DWC, 5x5_DWC, 7x7_DWC, 1x7_7x1_Conv, Id, 3x3_AvgPool, 3x3_MaxPool, 3x3_DilatedConv]
operations_indices = torch.arange(0,8)#number of operations considered is 8

#Have to transform them in tensors in order to use one hot encoding!

one_hot_inputs = torch.nn.functional.one_hot(inputs_indices)
one_hot_operations = torch.nn.functional.one_hot(operations_indices)

one_hot_inputs = one_hot_inputs.tolist()
one_hot_operations = one_hot_operations.tolist()

# one-hot --> labels
#labels_again = torch.argmax(one_hot_inputs, dim=1)
#labels_again = torch.argmax(one_hot_operations, dim=1)

def flatten(list_of_lists_of_tuples):
    """
    Utility function to make the list at each step a list of tuples. Each elemnt will be a cell, each tuple a block
    """
    x = []
    for elem in list_of_lists_of_tuples:
        tup = elem[0]+(elem[1],)#TOCHECK Add like this to obtain a tuple of tuples
        x.append(tup)
    #list_of_lists[0][0]
    return x#Return a list of tuples of tuples

def get_combinations(inputs_indices, operations_indices):
    """
    Get all the possible cells. By passing to the function a sliced inputs_indices, i
    can get all the possible cell structures!!!
    """
    #Obtain the combination with replacement of input indices
    #e.g. 'ABC' ----> 'AA', 'AB', 'AC', 'BB', 'BC', 'CC'
    x = list(itertools.combinations_with_replacement(inputs_indices, 2))

    #Obtain the pairs excluded in the previous list
    #e.g. 'ABC' ----> 'BA', 'CA', 'CB'
    xx = list(itertools.product(inputs_indices, repeat=2))
    xx = [elem for elem in xx if elem not in x]
    #xx = [elem for elem in xx if (torch.equal(xx[0][0], x[0][0]) and torch.equal(xx[0][1], x[0][1]))]

    #Obtain the combination with replacement of operations indices
    #e.g. '123' ----> '11', '12', '13', '22', '23', '33' 
    y = list(itertools.combinations_with_replacement(operations_indices, 2))

    #Obtain the combination of operations indices
    #e.g. '123' ----> '12', '13', '23'
    yy = list(itertools.combinations(operations_indices, 2))

    #Obtain the cartesian product
    final1 = list(itertools.product(x, y))
    final2 = list(itertools.product(xx, yy))
    
    return final1 + final2
    #From here probably I have to prepare the dataset and then the dataloader!!

def expand_cells(num_blocks, inputs_indices, operations_indices):#TOCHECK
    combinations = get_combinations(inputs_indices[:2+num_blocks-1], operations_indices)
    if num_blocks > 1:
        y = get_combinations(inputs_indices[:2+num_blocks], operations_indices)
        combinations = list(itertools.product(combinations, y))
    return combinations

def top_k(combinations, accuracies):
    return

def select_input(inputs_index, inputs_list):
    return 


#FIXME to see how the lists are passed; what i wrote is probably incorrect
def build_cell(cell_list,  num_filters = None):
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
    cell_template_list = []
    block_list = nn.ModuleList()
    for cell in cell_list:
        for block in cell:
            inputs_list.append([block[0][0].index(1), block[0][1].index(1)])
            #operations_list.append(block[1])
            block_list.append(m.Block_Template(block[1][0].index(1), block[1][1].index(1)))
        cell_template_list.append(m.Cell_Template(inputs_list, block_list))
        block_list = nn.ModuleList()

    print("-----------")
    return cell_template_list

#FIXME to see how the lists are passed; what i wrote is probably incorrect
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
        inputs_list.append([block[0][0].index(1), block[0][1].index(1)])
        #operations_list.append(block[1])
        block_list.append(m.Block_Template(block[1][0].index(1), block[1][1].index(1)))
    cell_ = m.Cell_Template(inputs_list, block_list, stride, num_filters)

    return cell_

def build_CIFAR10_model(cell, num_repetitions, num_filters, img_input_channels, num_classes):
    #--->take the inputs and the operations for block
    """for i in range(num_blocks):
        inputs = cell[0][i]#FIXME placeholder addresses
        operations = cell[1][i]
        inputs_list = []

        op1 = select_operation()
        op2 = select_operation()"""

    seq1 = nn.Sequential(*([build_cell2(cell, 1, num_filters=num_filters)]\
                                + [build_cell2(cell, 1) for _ in range(num_repetitions-1)]))
    
    seq2 = build_cell2(cell, 2)

    seq3 = nn.Sequential(*([build_cell2(cell, 1) for _ in range(num_repetitions)]))

    seq4 = build_cell2(cell, 2)

    seq5 = nn.Sequential(*([build_cell2(cell, 1) for _ in range(num_repetitions)]))
    #I have to extract from the last cell output,  that is  a list, the last output
    global_pool = nn.AdaptiveAvgPool2d((1, 1))

    linear = nn.LazyLinear(num_classes)#TOCHECK It says softmax classification layer

    model = nn.Sequential(seq1, seq2, seq3, seq4, seq5, global_pool, linear)

    return model

"""
------B == 1------
- Derive all possible combinations(get_combinations2(inputs_indices[:2], operations_indices) )

- Build the CNNs

- Train the Predictor<------HERE I NEED ONE HOT ENCODING

------B > 1------
- Expand cells:
    - Derive all possible combinations(e.g. b=1. get_combinations2(inputs_indices[:3], operations_indices)--->(last [:B]))
    - Combine with the previous cells!

- Predict Accuracy:<------HERE I NEED ONE HOT ENCODING
    - One hot encodings (is it alltogheter or are they two separate one hot encodings??)
    - Build the two dataloaders: One for Inputs, one for Operations
    - Train the predictor
    - Test on Validation

- Select top-K Cells

- Build the CNNs

- Train CNNs

- Fine Tune Predictor<------HERE I NEED ONE HOT ENCODING
"""


"""x = get_combinations(one_hot_inputs[:2], one_hot_operations)
#x = flatten2(x)
#----> da allenare

build_cell([x[128]])

y = get_combinations(one_hot_inputs[:3], one_hot_operations)
#y = flatten(y)

tr = list(itertools.product(x, y))
tr = tr[:K]
build_cell(tr)

w = get_combinations(one_hot_inputs[:4], one_hot_operations)

tr = list(itertools.product(tr, w))
tr = tr[:K]

tr = flatten(tr)
build_cell(tr[0])


z = get_combinations(one_hot_inputs[:5], one_hot_operations)

tr = list(itertools.product(tr, z))
tr = tr[:K]

tr = flatten(tr)

print("ciao")"""
