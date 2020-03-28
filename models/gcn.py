#!/usr/bin/env python
import torch
import torch.nn as nn

class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, A_norm, relu = None, 
                 featureless = False, dropout = 0.):
        super(GCNLayer, self).__init__()
        self.A_norm = A_norm
        self.featureless = featureless
        self.W = nn.Parameter(torch.randn(input_dim, output_dim))
        self.relu = nn.ReLU() if relu else None
        self.dropout = nn.Dropout(dropout)

        
    def forward(self, x):
        x = self.dropout(x)
        x = self.W if self.featureless else x.mm(self.W)
        out = self.A_norm.mm(x)
        
        if self.relu is not None:
            out = self.relu(out)

        self.embedding = out
        return out


class TextGCN(nn.Module):
    def __init__(self, input_dim, A_norm, 
                 dropout=0., num_classes=20):
        super(TextGCN, self).__init__()
        
        # GraphConvolution
        self.layer1 = GCNLayer(input_dim, 200, A_norm, relu=True, 
                              featureless=True, dropout=dropout)
        self.layer2 = GCNLayer(200, num_classes, A_norm, dropout=dropout)
        
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out
    