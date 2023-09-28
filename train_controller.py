import torch
from tqdm import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Function to train a model.
def train(controller, epochs, trainloader, optimizer, criterion):
    
    #telling wand to watch
    #if wandb.run is not None:
    #wandb.watch(model, optimizer, log="all", log_freq=320)

    controller.train()
    #model.load_state_dict(torch.load("/home/hsilva/lipreading/models/model_AV_500_4.pt"))
    # Training loop

    for epoch in range(epochs):

        progress_bar = tqdm(total=len(trainloader), unit='step')
        losses = []
        for cells, accuracies in trainloader:
            
            #
            cells = cells.to(device)
            accuracies = accuracies.to(device)

            optimizer.zero_grad()

            output = controller(cells)
            output = output.squeeze()

            loss = criterion(output, accuracies)
            loss.backward()
            
            optimizer.step()

            losses.append(loss.item())

            """#progress bar stuff
            progress_bar.set_description(f"Epoch {epoch+1}/{config.EPOCHS}")
            #progress_bar.set_postfix(loss=loss.item())  # Update the loss value
            progress_bar.set_postfix(loss=np.mean(losses))  # Update the loss value
            progress_bar.update(1)"""
        
        # endfor batch 
        
        #if wandb.run is not None:
        #wandb.log({"epoch":epoch, "loss":np.mean(losses)})
        
        # save the model
        #if epoch%1 == 0:
            #val_accuracy = test(model, valloader, vocabulary, ctc_loss)
            #wandb.log({"val_loss":val_accuracy})

        #if epoch%100 == 0:
            #torch.save(model.state_dict(), "models/model"+str(modeltitle)+"5.pt")
            

        #if epoch%1 == 0:
            #save_results(f"./results/results_{epoch}.txt", real_sentences, pred_sentences, overwrite=True)

    return controller

def evaluate(controller, valloader):
    controller.eval()

    real_sentences = []
    pred_sentences = []
    outputs = []
    with torch.no_grad():

        for cells in valloader:

            # move data to GPU!
            cells = cells[0].to(device)#FIXME Added the addressing, Don't know why "cells" is a list

            output = controller(cells)
            outputs.append(output)

    outputs = torch.cat(outputs)        

    return outputs