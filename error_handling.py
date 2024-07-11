import torch
import os
import train_cnn as TCNN
import gc
import model as M
import utils as U
import data as D

def load_cells_and_accuracies(dir_path,  block):
    
    cells = torch.load(os.path.join(dir_path, f"cells_predicted_block{block}.pt"))
    accuracies = torch.load(os.path.join(dir_path, f"accuracies_predicted_block{block}.pt"))

    return cells, accuracies

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

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

cells, accuracies = load_cells_and_accuracies("pnas/run3", 3)
cells = [(((cells[i][0][0].item(), cells[i][0][1].item()),(cells[i][0][2].item(), cells[i][0][3].item())),((cells[i][1][0].item(), cells[i][1][1].item()),(cells[i][1][2].item(), cells[i][1][3].item())),((cells[i][2][0].item(), cells[i][2][1].item()), (cells[i][2][2].item(), cells[i][2][3].item())))for i in range(cells.shape[0]) ]
top_k_cells, top_k_accuracies = U.order_cells_and_accuracies(cells, accuracies, 64)

trainset, testset, valset = D.get_CIFAR10(validation_size=5000)

U.set_reproducibility(10)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 512, shuffle=True, num_workers = 8)
valloader = torch.utils.data.DataLoader(valset, batch_size = 512, shuffle=False, num_workers = 8)
criterion_cnn = torch.nn.CrossEntropyLoss()

accuracies = train_cnn_models(top_k_cells, 24, 2, 20, criterion_cnn, trainloader, valloader, device)
accuracies = torch.tensor(accuracies)####
print(f"FINISHED BLOCK 3")

#Save the accuracies and the corresponding cells with num_blocks blocks
tensor_accuracies = torch.tensor(accuracies)
tensor_cells = U.cells_to_tensor(top_k_cells)
torch.save(tensor_cells, os.path.join("pnas/run3", f"cells_block3.pt"))
torch.save(tensor_accuracies, os.path.join("pnas/run3", f"accuracies_block3.pt"))