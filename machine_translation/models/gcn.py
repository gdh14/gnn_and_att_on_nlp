#!/usr/bin/env python
import torch
import torch.nn as nn

class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout = 0.2):
        """
        each layer has the following form of computation
        H = f(A * H * W)
        H: (b, seq len, ninp)
        A: (b, seq len, seq len)
        W: (ninp, nout)
        """
        super(GCNLayer, self).__init__()
        self.W = nn.Parameter(torch.randn(input_dim, output_dim))
        self.b = nn.Parameter(torch.randn(output_dim))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, A):
        """
        H = relu(A * x * W)
        x: (b, seq len, ninp)
        A: (b, seq len, seq len)
        W: (ninp, nout)
        """
        x = self.dropout(x)
        x = torch.bmm(A, x)  # x: (b, seq len, ninp)
        x = x.matmul(self.W) + self.b
        x = self.relu(x)
        return x
        
        
class GCNEncoder(nn.Module):
    def __init__(self, ninp, nembed, nhid, nlayers, dropout):
        super(GCNEncoder, self).__init__()
        self.nhid = nhid
        self.nlayers = nlayers
        self.embedding = nn.Embedding(ninp, nembed)
        assert(nlayers > 0)
        layers = [GCNLayer(nembed, nhid)] + [GCNLayer(nhid, nhid) for _ in range(nlayers-1)]
        self.layers = nn.ModuleList(layers)
        self.linear = nn.Linear(2*nhid, nhid)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, A):
        """
        x: (seq len, b)
        A: (b, seq len, seq len)
        """
        x = x.t()
        b = x.shape[0]
        x = self.embedding(x)  # x: (b, seq len, ninp)
        x = self.dropout(x)
        hidden = []
        for layer in self.layers:
            x = layer(x, A) 
            hidden.append(x[:,0,:])
            
        # pooling
        mean = x.mean(dim=1)
        maxm = x.max(dim=1)[0]
        x = torch.cat((mean, maxm), dim=1)
        out = self.linear(self.dropout(x))
        hidden = torch.stack(hidden)
        return out, hidden

        