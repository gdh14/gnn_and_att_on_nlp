import numpy as np
import torch
from torch import nn, LongTensor
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
from IPython.core.debugger import set_trace

class MyDataset(Dataset):

    def __init__(self, X, Y=None):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.Y is not None:
            return (self.X[idx], self.Y[idx])
        return (self.X[idx], None)

def pad(seq, seq_lengths, pad_after=True):
    max_seq_len = max(seq_lengths)
    seq_tensor = Variable(torch.zeros((len(seq), max_seq_len))).long()
    # pad input tensor
    for idx, seq in enumerate(seq):
        seq_len = seq_lengths[idx]
        if pad_after:
            seq_tensor[idx, :seq_len] = LongTensor(np.asarray(seq).astype(int))
        else: 
            # pad before
            seq_tensor[idx, max_seq_len-seq_len:] = LongTensor(np.asarray(seq).astype(int))
    return seq_tensor

def batchify(data):
    X, Y = tuple(map(list, zip(*data)))
    seq_lengths = LongTensor([len(x) for x in X])
    X = pad(X, seq_lengths, pad_after=True)
    Y = LongTensor(Y)
    return X, Y

def batchify_test(data):
    X, Y = tuple(map(list, zip(*data)))
    seq_lengths = LongTensor([len(x) for x in X])
    X = pad(X, seq_lengths, pad_after=True)
    return X, Y

def text_2_int_list(ls, vocab_dict):
    """ map list of strings to list of list of ints """
    result = []
    for t in ls:
        sent = t.split()
        ints = [vocab_dict[w] if w in vocab_dict else vocab_dict["<unk>"] for w in sent]
        result.append(ints)
    return result