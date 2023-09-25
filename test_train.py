import torch
import itertools
import models as M
import controller as C
import data as D

import train_controller as TC
import train_cnn as TCNN

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def flatten(list_of_lists):
    #FIXME
    """
    Utility function to make the list at each step a list of tuples. Each elemnt will be a cell, each tuple a block.
    The "list_of_lists" come in as:
        list_of_lists
            |
            |___cell____
                        |__elem__
                        |        |___block
                        |        |
                        |        |___block
                        |
                        |__block
    
    I want
        list_of_lists
            |
            |___cekk____
                        |___block
                        |
                        |___block
                        |
                        |___block    
    """
    x = []
    for elem in list_of_lists:
        tup = elem[0]+ elem[1] #TOCHECK Add like this to obtain a tuple of tuples
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
    final1 = [[elem] for elem in final1]
    final2 = list(itertools.product(xx, yy))
    final2 = [[elem] for elem in final2]

    
    return final1 + final2
    #From here probably I have to prepare the dataset and then the dataloader!!


def expand_cells(inputs_indices, operations_indices, num_blocks = 1, prev_combinations = None):#TOCHECK
    combinations = get_combinations(inputs_indices[:2+num_blocks-1], operations_indices)
    
    #If num_blocks > 1, we must have the list of previous valid combinations and make the cross product to 
    #obtain the new cell structures
    if num_blocks > 1:
        #y = get_combinations(inputs_indices[:2+num_blocks], operations_indices)
        combinations = list(itertools.product(prev_combinations, combinations))
        if num_blocks > 1:
            combinations = flatten(combinations)
    #TOCHECK If num_blocks == 1, I have to reformulate the list to be compatible with next functions
    """else:
        combinations = [[elem] for elem in combinations]"""
    
    return combinations

def cells_to_tensor(cells):
    tensor_cells = []
    for cell in cells:
        blocks = []
        for block in cell:
            inputs = torch.tensor(block[0]).view(-1)
            operations = torch.tensor(block[1]).view(-1)
            in_op = torch.cat((inputs, operations), dim=0)
            blocks.append(in_op[None, :])
        
        blocks = torch.cat(blocks)
        tensor_cells.append(blocks[None,:])

    tensor_cells = torch.cat(tensor_cells)
    
    return tensor_cells

def tensors_to_loader(tensors):
    """tensor_accuracies = torch.tensor(top_k_accuracies)
    tensor_cells = cells_to_tensor(top_k_cells)"""

    set = torch.utils.data.TensorDataset(*tensors)
    loader = torch.utils.data.DataLoader(set, batch_size = 128, shuffle=True)
    return

def _to_loader(cells, accuracies):
    tensor_accuracies = torch.tensor(accuracies)
    tensor_cells = cells_to_tensor(cells)

    return tensors_to_loader((tensor_cells, tensor_accuracies))

"""
------B == 1------
- Derive all possible combinations(get_combinations2(inputs_indices[:2], operations_indices) )

- Build the CNNs

- Train CNNs

- Save Validation Accuracies

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

- Save Validation Accuracies

- Fine Tune Predictor, with new accuracies<------HERE I NEED ONE HOT ENCODING
"""
B = 5 #Maximum number of blocks
K = 256#Number of Model samples
N = 2#Number of repetition of the cells with stride 1
F = 24#Number of filters in the forst laye
EPOCHS = 1
EPOCHS_CONTROLLER = 10

#[H_1, ..., H_(B-1), H_prev_cell, H_(prev_cell-1)]
inputs_indices = list(range(2+B-1))#2 previous cell, B-1 as the last block can also have B-1 possible inputs from previous blocks in the same cell
#inputs_indices = list(string.ascii_uppercase)[:2+B-1]

#[3x3_DWC, 5x5_DWC, 7x7_DWC, 1x7_7x1_Conv, Id, 3x3_AvgPool, 3x3_MaxPool, 3x3_DilatedConv]
operations_indices = list(range(8))#number of operations considered is 8

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

tr = flatten(tr)"""

"""
def ProgressiveNeuralArchitectureSearch(max_num_blocks, num_filters_first_layer, beam_size, num_times_unroll_cell, max_epochs, trainset, valset):
    cells = expand_cells(one_hot_inputs, one_hot_operations, 1)
    model_list = []
    for cell in cells:
        model_list.append(M.Full_Convolutional_Template(cell, num_times_unroll_cell, num_filters_first_layer, num_classes=10))

    accuracies = []
    for model in model_list:
        model = train(model, max_epochs, trainset)
        accuracies.append(validate(model, valset))
    
    controller = C.RNN_controller()

    #I HAVE TO CREATE THE DATA LOADER!! FOR CELLS AND ACCURACIES
    #TO TENSOR AND THEN TO DATALOADER??

    controller = train(controller, cells, accuracies)

    for num_blocks in range(2, max_num_blocks+1):
        cells = expand_cells(one_hot_inputs, one_hot_operations, num_blocks)
        
        
        ordered_predictor_accuracies = []
        ordered_cells = []
        temp_acc = 0
        for index, cell in enumerate(cells):
            acc = validate(controller, cell)
            if temp_acc < acc:
                temp_acc = acc
                ordered_cells.insert(0, cell)
                ordered_predictor_accuracies.insert(0, acc)
            else:
                ordered_cells.append(cell)
                ordered_predictor_accuracies.append(acc)

        
        top_k_cells = ordered_cells[:beam_size]
        top_k_accuracies = ordered_predictor_accuracies[:beam_size]

        model_list = []
        for cell in top_k_cells:
            model_list.append(M.Full_Convolutional_Template(cell, num_times_unroll_cell, num_filters_first_layer, num_classes=10))

        accuracies = []
        for model in model_list:
            model = train(model, max_epochs, trainset)
            accuracies.append(validate(model, valset))

        #I HAVE TO CREATE THE DATA LOADER!! FOR CELLS AND ACCURACIES
        #TO TENSOR AND THEN TO DATALOADER??

        controller = train(controller, cells, accuracies)#HERE IS FINETUNING!!!

    
    return top_k_cells[0]
"""

"""mod = C.RNN_controller(6, 8, 100, 100, 2)

cells = []
prev_cells = None
for i in range(1,6):
    cells = expand_cells(inputs_indices, operations_indices, i, prev_cells)
    cells = cells[:50]
    prev_cells = cells
    #transform cells in tensors:
    tensor_cells = cells_to_tensor(cells)
    out = mod(tensor_cells)"""

def ProgressiveNeuralArchitectureSearch(input_indices, operations_indices, max_num_blocks, num_filters_first_layer, beam_size, num_times_unroll_cell, max_epochs, trainloader, valloader):

    #TODO Move this from here
    num_input_indices = len(input_indices)
    num_operations_indices = len(operations_indices)
    controller = C.RNN_controller(num_input_indices, num_operations_indices, 100, 100, 2).to(device)#TOCHECK see if is better to create this outside
    optimizer_contr = torch.optim.Adam(controller.parameters(), 0.01)
    criterion_contr = torch.nn.L1Loss()

    criterion_cnn = torch.nn.CrossEntropyLoss()

    cells = expand_cells(input_indices, operations_indices, 1)
    model_list = []
    for cell in cells:
        model_list.append(M.Full_Convolutional_Template(cell, num_times_unroll_cell, num_filters_first_layer, num_classes=10))

    accuracies = []
    for model in model_list[:2]:#DEBUG
        optimizer_cnn = torch.optim.Adam(model.parameters(), 0.01)
        #TODO Control what T_max really is
        scheduler_cnn = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_cnn, T_max = 50)
        #TOCHECK See if we hace to reset in some way the loaders!
        model = model.to(device)
        model = TCNN.train(model, max_epochs, trainloader, optimizer_cnn, scheduler_cnn, criterion_cnn)
        accuracies.append(TCNN.evaluate(model, valloader, criterion_cnn))

    #I HAVE TO CREATE THE DATA LOADER!! FOR CELLS AND ACCURACIES
    #TO TENSOR AND THEN TO DATALOADER??
    tensor_accuracies = torch.tensor(accuracies)
    tensor_cells = cells_to_tensor(cells[:2])#DEBUG
    trainlset_contr = torch.utils.data.TensorDataset(tensor_cells, tensor_accuracies)
    trainloader_contr = torch.utils.data.DataLoader(trainlset_contr, batch_size = 128, shuffle=True)

    controller = TC.train(controller, 1, trainloader_contr, optimizer_contr, criterion_contr)
    #Change the learning rate with number of blocks > 1
    #TOCHECK is this the correct way???
    for g in optimizer_contr.param_groups:
        g['lr'] = 0.002

    #I need this in the for cycle below to update the cell structure
    #TOCHECK maybe I can use the cells variable directly
    prev_cells = cells

    for num_blocks in range(2, max_num_blocks+1):
        cells = expand_cells(input_indices, operations_indices, num_blocks, cells)
        ordered_predictor_accuracies = []
        ordered_cells = []
        temp_acc = 0
        tensor_cells = cells_to_tensor(cells)
        valset_contr = torch.utils.data.TensorDataset(tensor_cells)#FIXME Had to add the *, as it returns tensor_cells is a list
        valloader_contr = torch.utils.data.DataLoader(valset_contr, batch_size = 128, shuffle=False)

        acc = TC.evaluate(controller, valloader_contr)
        for index, cell in enumerate(cells):
            if temp_acc < acc[index]:
                temp_acc = acc[index]
                ordered_cells.insert(0, cell)
                ordered_predictor_accuracies.insert(0, acc[index])
            else:
                ordered_cells.append(cell)
                ordered_predictor_accuracies.append(acc[index])

        
        top_k_cells = ordered_cells[:beam_size]
        top_k_accuracies = ordered_predictor_accuracies[:beam_size]

        model_list = []
        for cell in top_k_cells:
            model_list.append(M.Full_Convolutional_Template(cell, num_times_unroll_cell, num_filters_first_layer, num_classes=10))

        accuracies = []
        for model in model_list:
            optimizer_cnn = torch.optim.Adam(model.parameters(), 0.01)
            #TODO Control what T_max really is
            scheduler_cnn = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_cnn, T_max = 50)
            #TOCHECK See if we hace to reset in some way the loaders!
            model = model.to(device)
            model = TCNN.train(model, max_epochs, trainloader, optimizer_cnn, scheduler_cnn, criterion_cnn)
            #TOCHECK maybe here I have to put the testloader and not the valloader
            accuracies.append(TCNN.evaluate(model, valloader, criterion_cnn))

        #I HAVE TO CREATE THE DATA LOADER!! FOR CELLS AND ACCURACIES
        #TO TENSOR AND THEN TO DATALOADER??
        
        tensor_accuracies = torch.tensor(top_k_accuracies)
        tensor_cells = cells_to_tensor(top_k_cells)
        trainlset_contr = torch.utils.data.TensorDataset(tensor_cells, tensor_accuracies)
        trainloader_contr = torch.utils.data.DataLoader(trainlset_contr, batch_size = 128, shuffle=True)
        controller = TC.train(controller, 1, trainloader_contr, optimizer_contr, criterion_contr)#HERE IS FINETUNING!!!
        cells = top_k_cells

    #ENDFOR

    return top_k_cells[0]



trainset, valset, testset = D.get_CIFAR10(5000)

trainloader = torch.utils.data.DataLoader(trainset, batch_size = 128, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size = 128, shuffle=False)


ProgressiveNeuralArchitectureSearch(inputs_indices, operations_indices, B, F, K, N, EPOCHS, trainloader, valloader)







