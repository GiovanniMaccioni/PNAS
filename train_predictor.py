import torch
from tqdm import tqdm
import numpy as np



# Function to train a model.
def train(predictor, epochs, trainloader, optimizer, criterion, device):
    predictor.train()

    for epoch in range(epochs):

        progress_bar = tqdm(total=len(trainloader), unit='step')
        losses = []
        for cells, accuracies in trainloader:
            #
            cells = cells.to(device)
            accuracies = accuracies.to(device)

            optimizer.zero_grad()

            output = predictor(cells)
            output = output.squeeze()

            loss = criterion(output, accuracies)
            loss.backward()
            
            optimizer.step()

            losses.append(loss.item())

            #progress bar stuff
            progress_bar.set_description(f"Predictor Epoch {epoch}")
            progress_bar.set_postfix(loss=np.mean(losses))  # Update the loss value
            progress_bar.update(1)
        
        # endfor batch

    return predictor

def evaluate(predictor, valloader, device):
    predictor.eval()
    outputs = []
    with torch.no_grad():

        for cells in valloader:
            # move data to GPU!
            cells = cells.to(device)
            output = predictor(cells)
            outputs.append(output)

    outputs = torch.cat(outputs)        

    return outputs