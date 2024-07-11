import os
import torch
import predictor as C
import train_predictor as TC
import random
import numpy as np
import utils as U

def train_predictor(accuracies, cells, predictor, optimizer_pr, criterion_pr, epochs, device):

    #Create a Dataloader for Predictor training
    trainlset_pr = torch.utils.data.TensorDataset(cells, accuracies)
    trainloader_pr = torch.utils.data.DataLoader(trainlset_pr, batch_size = 64, shuffle=True)

    #Train/Finetune the Predictor
    predictor = TC.train(predictor, epochs, trainloader_pr, optimizer_pr, criterion_pr, device)

    return predictor

def evaluate_predictor(cells, predictor, device):
    valset_pr = cells
    valloader_pr = torch.utils.data.DataLoader(valset_pr, batch_size = 64, shuffle=False)
    accuracies_predicted = TC.evaluate(predictor, valloader_pr, device)

    return  accuracies_predicted

def train_predictor_ensemble(accuracies, cells, predictor, optimizer_pr, criterion_pr, epochs, device):
    
    #Find the number of slices to train the components of the model ensemble with 4/5 of the cells 
    slice = len(cells)//5
    rest = len(cells)%5
    indices = list(range(0, len(cells)))
    random.shuffle(indices)

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
        trainlset_pr = torch.utils.data.TensorDataset(cells, accuracies)
        trainloader_pr = torch.utils.data.DataLoader(trainlset_pr, batch_size = 64, sampler=torch.utils.data.RandomSampler(ind))
        predictor[i] = TC.train(predictor[i], epochs, trainloader_pr, optimizer_pr[i], criterion_pr, device)

    return predictor

def evaluate_predictor_ensemble(cells, predictor, device):

    #Create the Dataloader to evaluate the performance of cells with b blocks(not yet seen)
    valset_pr = cells
    valloader_pr = torch.utils.data.DataLoader(valset_pr, batch_size = 64, shuffle=False)

    accuracies_predicted = []
    #Using the trained/finetuned ensemble, predict the performance of the cell structures with b blocks for each component
    for i in range(5):
        acc = TC.evaluate(predictor[i], valloader_pr, device)
        accuracies_predicted.append(acc)
        #accumulate accuracies make a mean and have the result
    
    #Perform the mean accuracy of the predictions of the ensemble components
    accuracies_predicted = torch.mean(torch.cat(accuracies_predicted, dim=1), dim=1)[:, None]

    return accuracies_predicted

def load_cells_and_accuracies(dir_path,  block):
    
    cells = torch.load(os.path.join(dir_path, f"cells_block{block}.pt"))
    accuracies = torch.load(os.path.join(dir_path, f"accuracies_block{block}.pt"))

    return cells, accuracies

def correlation_(len_inputs, num_operations, num_blocks, k, T, epochs_pr, dir_path, ensemble, device):

    criterion_pr = torch.nn.L1Loss()
    #
    b_accuracies_selected_list = []
    b_accuracies_predicted_list = []
    bp1_accuracies_selected_list = []
    bp1_accuracies_predicted_list = []

    if ensemble:
            predictor = torch.nn.ModuleList([C.MLP(len_inputs, num_operations, 100, 100).to(device) for _ in range(5)])
            optimizer_pr = [torch.optim.Adam(predictor[i].parameters(), 0.01) for i in range(5)]
    else:
        predictor = C.MLP(len_inputs, num_operations, 100, 100).to(device)
        optimizer_pr = torch.optim.Adam(predictor.parameters(), 0.01)

    
    for b in range(1,num_blocks):
        b_accuracies_selected_list.append([])
        b_accuracies_predicted_list.append([])
        bp1_accuracies_selected_list.append([])
        bp1_accuracies_predicted_list.append([])

        if b == 2:
            if ensemble:
                for i in range(5):
                    for g in optimizer_pr[i].param_groups:
                        g['lr'] = 0.002
            else:
                for g in optimizer_pr.param_groups:
                    g['lr'] = 0.002

        for t in range(T):

            cells, accuracies = load_cells_and_accuracies(dir_path, b)
            #sample k cells from the cells
            indices = list(np.random.choice(len(cells), (k if k < len(cells) else len(cells)), replace=False))
            cells_selected = [cells[i][None, :] for i in indices]
            accuracies_selected = [accuracies[i][None] for i in indices]

            cells_selected = torch.cat(cells_selected, dim=0)
            accuracies_selected = torch.cat(accuracies_selected, dim=0)

            if ensemble:
                predictor = train_predictor_ensemble(accuracies_selected, cells_selected, predictor, optimizer_pr, criterion_pr, epochs_pr, device)
                accuracies_predicted = evaluate_predictor_ensemble(cells_selected, predictor, device)
            else:
                predictor = train_predictor(accuracies_selected, cells_selected, predictor, optimizer_pr, criterion_pr, epochs_pr, device)
                accuracies_predicted = evaluate_predictor(cells_selected, predictor, device)

            b_accuracies_selected_list[b-1].append(accuracies_selected)
            b_accuracies_predicted_list[b-1].append(accuracies_predicted.squeeze())

            cells, accuracies = load_cells_and_accuracies(dir_path, b+1)

            if ensemble:
                accuracies_predicted = evaluate_predictor_ensemble(cells, predictor, device)
            else:
                accuracies_predicted = evaluate_predictor(cells, predictor, device)
            
            bp1_accuracies_selected_list[b-1].append(accuracies)
            bp1_accuracies_predicted_list[b-1].append(accuracies_predicted.squeeze())
            
        
    #PLOTTING
    for b in range(num_blocks-1):
        x = b_accuracies_selected_list[b]
        y = b_accuracies_predicted_list[b]
        x = torch.cat(x).cpu().numpy()
        y = torch.cat(y).cpu().numpy()
        U.plot_correlationplot(y, x, 'Predicted Score', 'Training Score', os.path.join(dir_path, f"current_level_b{b+1}.png"))

        x = bp1_accuracies_selected_list[b]
        y = bp1_accuracies_predicted_list[b]
        x = torch.cat(x).cpu().numpy()
        y = torch.cat(y).cpu().numpy()
        U.plot_correlationplot(y, x, 'Predicted Score', 'Training Score', os.path.join(dir_path, f"next_level{b+2}.png"))

    return