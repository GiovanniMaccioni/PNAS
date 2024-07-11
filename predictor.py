#For now here there will be the all the code for the predictor

import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    """
    def __init__(self, input_dim_I, input_dim_O, emb_dim, hid_units):
        super().__init__()
        self.embedding_I = nn.Embedding(input_dim_I, emb_dim)
        self.embedding_O = nn.Embedding(input_dim_O, emb_dim)

        self.embedding_I.weight.data.uniform_(-0.1, 0.1)
        self.embedding_O.weight.data.uniform_(-0.1, 0.1)

        self.linear1 = nn.Linear(4*emb_dim, hid_units)
        self.linear2 = nn.Linear(hid_units, hid_units)
        self.linear3 = nn.Linear(hid_units, 1)

        self.actv = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):

        emb_I1 = self.embedding_I(x[:, :, 0])#[batch_size, num_blocks, emb_dim]
        emb_I2 = self.embedding_I(x[:, :, 1])#[batch_size, num_blocks, emb_dim]
        emb_O1 = self.embedding_O(x[:, :, 2])#[batch_size, num_blocks, emb_dim]
        emb_O2 = self.embedding_O(x[:, :, 3])#[batch_size, num_blocks, emb_dim]

        x = torch.cat((emb_I1, emb_I2, emb_O1, emb_O2), dim=(2))
        x = torch.mean(x, dim=(1)).squeeze()

        x = self.linear1(x)
        x = self.actv(x)
        x = self.linear2(x)
        x = self.actv(x)
        x = self.linear3(x)
        x = self.sig(x)

        return x
    




    
