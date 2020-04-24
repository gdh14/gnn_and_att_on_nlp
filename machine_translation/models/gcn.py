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
    
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)