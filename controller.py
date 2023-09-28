#For now here there will be the all the code for the controller

import torch
import torch.nn as nn

class RNN_controller(nn.Module):
    """
    input_dim_I: dimension of the one hot encoded vector of the input----> it changes dimension as the number of blocks increases??
    input_dim_O: dimension of the one hot encoded vector of the operations
    """
    def __init__(self, input_dim_I, input_dim_O, emb_dim, hid_dim, n_layers):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        #We have two different embedings for the inputs and the operations
        #TODO add uniform initialization between [-0.1,-0.1]
        self.embedding_I = nn.Embedding(input_dim_I, emb_dim)
        self.embedding_O = nn.Embedding(input_dim_O, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True)

        self.linear = nn.Linear(hid_dim, 1)#TOCHECK maybe it is an MLP as it talks about a fully connected layer
        self.sig = nn.Sigmoid()
        
    def forward(self, blocks):#blocks [number_of_cells, num_blocks, 28]--> for now i suppose to have a fixed one hot encoding length
        """
        Here i have to embed each element of the sequence in input with its specific embedding layer.
        The sequence is "4b" long, (I1, I2, O1, O2)*num_blocks.
        
        We take as input to the controller forward method two sequences; inputs and operations
        Then we embed them and concatenate them to have I1, I2, O1, O2 sequence
        of length 4*num_blocks

        Then we pass it to the LSTM and regress the accuracy
        """

        emb_I1 = self.embedding_I(blocks[:, :, 0])[:, :, None]#[batch_size, num_blocks, 1, emb_dim]
        emb_I2 = self.embedding_I(blocks[:, :, 1])[:, :, None]#[batch_size, num_blocks, 1, emb_dim]
        emb_O1 = self.embedding_O(blocks[:, :, 2])[:, :, None]#[batch_size, num_blocks, 1, emb_dim]
        emb_O2 = self.embedding_O(blocks[:, :, 3])[:, :, None]#[batch_size, num_blocks, 1, emb_dim]

        new_batch = torch.cat((emb_I1, emb_I2, emb_O1, emb_O2), dim=2)
        new_batch = torch.reshape(new_batch, (new_batch.shape[0], new_batch.shape[1]*new_batch.shape[2], new_batch.shape[3]))

        _, (hid, cell) = self.rnn(new_batch)

        out = self.linear(torch.mean(hid, dim=0))
        out = self.sig(out)

        return out#32x1?????




class MLP(nn.Module):
    """
    """
    #TOCHECK There is a bias term to add!!!!
    def __init__(self, input_dim_I, input_dim_O, emb_dim, hid_units):
        super().__init__()
        #TODO add uniform initialization between [-0.1,-0.1]
        #TOCHECK For now i supposed the same kind of embedding as the LSTM controller. But the paper
        #seems to hint to a unique embedding for both the input and operations
        self.embedding_I = nn.Embedding(input_dim_I, emb_dim)
        self.embedding_O = nn.Embedding(input_dim_O, emb_dim)

        self.linear1 = nn.Linear(emb_dim, hid_units)
        self.linear2 = nn.Linear(hid_units, hid_units)#FIXME bias = 1.8 (page 8)
        self.actv = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.actv(x)
        x = self.linear2(x)
        x = self.sig(x)

        return x

class EnsembleMLP_controller(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_units, num_ensembles):
        super().__init__()
        self.mlp_ens = nn.ModuleList([MLP(input_dim, emb_dim, hid_units) for _ in range(num_ensembles)])

    def forward(self, x):
        sum = torch.zeros((x.shape[0], 1))
        for mlp in self.mlp_ens:
            y = mlp(x)
            sum = sum + y
        
        return sum/x.shape[0] #return the average of the mlps output
    




    
