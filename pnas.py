import torch
import model as M
import predictor as C
import data as D
import utils as U
import train_predictor as TC
import train_cnn as TCNN

import gc
import os
import random
import numpy as np


def train_cnn_models(cells, num_filters, num_repetitions, epochs, criterion_cnn, trainloader, valloader, device):
    """
    Given a list of cells, instantiate and train the corresponding full convolutional networks
    
    output:
        Accuracies for all the models on the validation set
    """
    accuracies = []
    for count, cell in enumerate(cells):
        model = M.Full_Convolutional_Template(cell, 3, num_filters, len(cell), num_repetitions, 10)
        print(f"CNN #{count}: ", sum(p.numel() for p in model.parameters()))

        optimizer_cnn = torch.optim.Adam(model.parameters(), 0.01)
        scheduler_cnn = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_cnn, T_max = 20)

        model = model.to(device)
        model = TCNN.train(model, epochs, trainloader, optimizer_cnn, scheduler_cnn, criterion_cnn, device)
        accuracies.append(TCNN.evaluate(model, valloader, device))

        #Delete the model from GPU memory
        model = model.to("cpu")
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    return accuracies



def train_evaluate_predictor(top_k_accuracies, top_k_cells, predictor, optimizer_pr, criterion_pr, input_indices, operation_indices, b, epochs, dir_path, device):
    tensor_accuracies = torch.tensor(top_k_accuracies)
    tensor_cells = U.cells_to_tensor(top_k_cells)

    #Save the accuracies and the corresponding cells with b-1 blocks
    torch.save(tensor_cells, os.path.join(dir_path, f"cells_block{b-1}.pt"))
    torch.save(tensor_accuracies, os.path.join(dir_path, f"accuracies_block{b-1}.pt"))

    #Create a Dataloader for Predictor training
    trainlset_pr = torch.utils.data.TensorDataset(tensor_cells, tensor_accuracies)
    trainloader_pr = torch.utils.data.DataLoader(trainlset_pr, batch_size = 128, shuffle=True)

    #Train/Finetune the Predictor
    predictor = TC.train(predictor, epochs, trainloader_pr, optimizer_pr, criterion_pr, device)
    
    #From the best cells with b-1 blocks, create all the possible cells with b blocks
    cells = U.expand_cells(input_indices, operation_indices, b, top_k_cells)

    #Create the Dataloader to evaluate the performance of cells with b blocks(not yet seen)
    valset_pr = U.cells_to_tensor(cells)
    valloader_pr = torch.utils.data.DataLoader(valset_pr, batch_size = 128, shuffle=False)

    #Using the trained/finetuned predictor, predict the performance of the cell structures with b blocks
    accuracies = TC.evaluate(predictor, valloader_pr, device)

    #Save the accuracies and the corresponding cells with b blocks
    torch.save(accuracies, os.path.join(dir_path, f"accuracies_predicted_block{b}.pt"))
    tensor_cells = U.cells_to_tensor(cells)
    torch.save(tensor_cells, os.path.join(dir_path, f"cells_predicted_block{b}.pt"))

    return accuracies, cells

def train_evaluate_predictor_ensemble(top_k_accuracies, top_k_cells, predictor, optimizer_pr, criterion_pr, input_indices, operation_indices, b, epochs, dir_path, device):
    
    #Find the number of slices to train the components of the model ensemble with 4/5 of the cells 
    slice = len(top_k_cells)//5
    rest = len(top_k_accuracies)%5
    indices = list(range(0, len(top_k_cells)))
    random.shuffle(indices)

    tensor_accuracies = torch.tensor(top_k_accuracies)
    tensor_cells = U.cells_to_tensor(top_k_cells)

    #Save the accuracies and the corresponding cells with b-1 blocks
    torch.save(tensor_cells, os.path.join(dir_path, f"cells_block{b-1}.pt"))
    torch.save(tensor_accuracies, os.path.join(dir_path, f"accuracies_block{b-1}.pt"))

    #Find the indices to pass to the RandomSampler
    for i in range(5):
        if rest == 0:
            ind = indices[(i)*slice:(i+1)*slice]
            ind = [e for e in indices if e not in ind]
        else:
            ind = indices[i*slice:(i+1)*slice+1]
            ind = [e for e in indices if e not in ind]
            rest = rest - 1

        #Found the correct indices, Create the dataloader to train each component of the ensemble
        trainlset_pr = torch.utils.data.TensorDataset(tensor_cells, tensor_accuracies)
        trainloader_pr = torch.utils.data.DataLoader(trainlset_pr, batch_size = 128, sampler=torch.utils.data.RandomSampler(ind))
        predictor[i] = TC.train(predictor[i], epochs, trainloader_pr, optimizer_pr[i], criterion_pr, device)

    #From the best cells with b-1 blocks, create all the possible cells with b blocks
    cells = U.expand_cells(input_indices, operation_indices, b, top_k_cells)

    #Create the Dataloader to evaluate the performance of cells with b blocks(not yet seen)
    valset_pr = U.cells_to_tensor(cells)
    valloader_pr = torch.utils.data.DataLoader(valset_pr, batch_size = 128, shuffle=False)

    accuracies = []

    #Using the trained/finetuned ensemble, predict the performance of the cell structures with b blocks for each component
    for i in range(5):
        acc = TC.evaluate(predictor[i], valloader_pr, device)
        accuracies.append(acc)
        #accumulate accuracies make a mean and have the result
    
    #Perform the mean accuracy of the predictions of the ensemble components
    accuracies = torch.mean(torch.cat(accuracies, dim=1), dim=1)[:, None]

    #Save the accuracies and the corresponding cells with b blocks
    torch.save(accuracies, os.path.join(dir_path, f"accuracies_predicted_block{b}.pt"))
    tensor_cells = U.cells_to_tensor(cells)
    torch.save(tensor_cells, os.path.join(dir_path, f"cells_predicted_block{b}.pt"))

    return accuracies, cells




def ProgressiveNeuralArchitectureSearch(num_operations, num_blocks, num_filters, beam_size, num_repetitions, epochs_cnn, trainloader, valloader, ensemble, epochs_pr, dir_path, device):

    #Possible input indices;
    #2 are the previous cell and previous previous cell output
    #num_blocks-1 are the possible number of outputs from previous blocks, given the number of blocks num_blocks
    input_indices = list(range(2 + num_blocks-1))

    #Possible operation indices
    #[3x3_DWC, 5x5_DWC, 7x7_DWC, 1x7_7x1_Conv, Id, 3x3_AvgPool, 3x3_MaxPool, 3x3_DilatedConv]
    operation_indices = list(range(num_operations))
    if ensemble:
        predictor = torch.nn.ModuleList([C.MLP(len(input_indices), num_operations, 100, 100).to(device) for _ in range(5)])
        optimizer_pr = [torch.optim.Adam(predictor[i].parameters(), 0.01) for i in range(5)]
    else:
        predictor = C.MLP(len(input_indices), num_operations, 100, 100).to(device)
        optimizer_pr = torch.optim.Adam(predictor.parameters(), 0.01)

    criterion_pr = torch.nn.L1Loss()

    criterion_cnn = torch.nn.CrossEntropyLoss()

    #Create the list of cells with 1 block
    cells = U.expand_cells(input_indices, operation_indices, 1)
    #DEBUG
    #cells = cells[:5]

    for b in range(1, num_blocks):
        
        #Given the cells with b blocks, train the corresponding CNNs and compute the accuracies on the Cifar validation set
        accuracies = train_cnn_models(cells, num_filters, num_repetitions, epochs_cnn, criterion_cnn, trainloader, valloader, device)
        
        #Given the cells and corresponding accuracies,  train/finetune the predictor and predict the accuracies on unseen cells with b+1 blocks
        if ensemble:
            accuracies, cells = train_evaluate_predictor_ensemble(accuracies, cells, predictor, optimizer_pr, criterion_pr, input_indices, operation_indices, b+1, epochs_pr, dir_path, device)
        else:
            accuracies, cells = train_evaluate_predictor(accuracies, cells, predictor, optimizer_pr, criterion_pr, input_indices, operation_indices, b+1, epochs_pr, dir_path, device)

        print(f"FINISHED EVALUATING BLOCK {b+1}")

        #Given the evaluated cells with b+1 blocks, 
        top_k_cells, top_k_accuracies = U.order_cells_and_accuracies(cells, accuracies, beam_size)
        cells = top_k_cells

        #
        if b == 1:
            if ensemble:
                for i in range(5):
                    for g in optimizer_pr[i].param_groups:
                        g['lr'] = 0.002
            else:
                for g in optimizer_pr.param_groups:
                    g['lr'] = 0.002

        print(f"FINISHED BLOCK {b}")


    ##Given the cells with num_blocks blocks, train the corresponding CNNs and compute the accuracies on the Cifar validation set
    accuracies = train_cnn_models(cells, num_filters, num_repetitions, epochs_cnn, criterion_cnn, trainloader, valloader, device)
    accuracies = torch.tensor(accuracies)####
    top_k_cells, top_k_accuracies = U.order_cells_and_accuracies(cells, accuracies, beam_size)

    print(f"FINISHED BLOCK {num_blocks}")

    #Save the accuracies and the corresponding cells with num_blocks blocks
    tensor_accuracies = torch.tensor(top_k_accuracies)
    tensor_cells = U.cells_to_tensor(top_k_cells)
    torch.save(tensor_cells, os.path.join(dir_path, f"cells_block{num_blocks}.pt"))
    torch.save(tensor_accuracies, os.path.join(dir_path, f"accuracies_block{num_blocks}.pt"))#accuracies_predicted_block

    return top_k_cells[0], top_k_accuracies[0]

def train_random_models(num_operations, num_blocks, num_filters, num_samples, num_repetitions, epochs_cnn, trainloader, valloader, dir_path, device):
    #Possible input indices;
    #2 are the previous cell and previous previous cell output
    #num_blocks-1 are the possible number of outputs from previous blocks, given the number of blocks num_blocks
    input_indices = list(range(2 + num_blocks-1))

    #Possible operation indices
    #[3x3_DWC, 5x5_DWC, 7x7_DWC, 1x7_7x1_Conv, Id, 3x3_AvgPool, 3x3_MaxPool, 3x3_DilatedConv]
    operation_indices = list(range(num_operations))

    #Take all the cells with num_block blocks
    cells = U.expand_cells(input_indices, operation_indices, 1)
    for b in range(2, num_blocks + 1):
        cells = U.expand_cells(input_indices, operation_indices, b, cells)
    
    #Uniformly sample num_samples samples from the cells
    indices = list(np.random.choice(len(cells), (num_samples if num_samples < len(cells) else len(cells)), replace=False))
    cells = [cells[i] for i in indices]

    criterion_cnn = torch.nn.CrossEntropyLoss()

    #train and evaluate the corresponding CNNs
    accuracies = train_cnn_models(cells, num_filters, num_repetitions, epochs_cnn, criterion_cnn, trainloader, valloader, device)

    #Save the accuracies and the corresponding cells
    tensor_accuracies = torch.tensor(accuracies)
    tensor_cells = U.cells_to_tensor(cells)
    torch.save(tensor_cells, os.path.join(dir_path, f"cells_block{num_blocks}.pt"))
    torch.save(tensor_accuracies, os.path.join(dir_path, f"accuracies_block{num_blocks}.pt"))

    return

def run(c):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    B = c['num_blocks'] #Maximum number of blocks
    K = c['k']#256 Number of Model samples
    N = c['num_repetitions']#Number of repetition of the cells with stride 1
    F = c['num_filters']#Number of filters in the first layer
    NUM_OP = 8

    EPOCHS = 20
    EPOCHS_predictor = 5

    if not os.path.exists(c['dir_path']):
        os.makedirs(c['dir_path'])

    if (c['pnas']):
        with open(os.path.join(c['dir_path'], "config.txt"), 'w') as f:
            print(c, file=f)
    else:
        with open(os.path.join(c['dir_path'], f"config_block{B}.txt"), 'w') as f:
            print(c, file=f)

    trainset, testset, valset = D.get_CIFAR10(validation_size=5000)

    U.set_reproducibility(c['seed'])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = c['batch_size_cifar'], shuffle=True, num_workers = 8)
    valloader = torch.utils.data.DataLoader(valset, batch_size = c['batch_size_cifar'], shuffle=False, num_workers = 8)

    if (c['pnas']):
        top, _ = ProgressiveNeuralArchitectureSearch(NUM_OP, B, F, K, N, EPOCHS, trainloader, valloader, True, EPOCHS_predictor, c['dir_path'], device)
        print(top)
    else:
        train_random_models(NUM_OP, B, F, K, N, EPOCHS, trainloader, valloader, c['dir_path'], device)







