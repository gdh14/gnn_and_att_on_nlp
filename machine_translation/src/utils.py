import spacy, random, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext.datasets import TranslationDataset, Multi30k, IWSLT
from torchtext.data import Field, BucketIterator, RawField, Dataset
import datetime, os


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
# Tokenization
    
spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

def tokenize_de(text, reverse=False):
    """
    Tokenizes German text from a string into a list of strings (tokens) and reverses it
    """
    tokens = [tok.text for tok in spacy_de.tokenizer(text)]
    return tokens[::-1] if reverse else tokens
    
def tokenize_en(text, reverse=False):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    tokens = [tok.text for tok in spacy_en.tokenizer(text)]
    return tokens[::-1] if reverse else tokens
    
def batch_graph(grhs):
    """ batch a list of graphs
    @param grhs: list(tensor,...) 
    """
    b = len(grhs)  # batch size
    graph_dims = [len(g) for g in grhs]
    s = max(graph_dims)  # max seq length
    
    G = torch.zeros([b, s, s])
    for i, g in enumerate(grhs):
        s_ = graph_dims[i]
        G[i,:s_,:s_] = g
    return G

# Sentence data

from collections import Counter

def get_sentence_lengths(dataset):
    src_counter = Counter()
    tgt_counter = Counter()
    for exp in dataset:
        src_counter[len(exp.src)] += 1
        tgt_counter[len(exp.trg)] += 1
    return src_counter, tgt_counter

def counter2array(counter):
    result = []
    for k in counter:
        result.extend([k for _ in range(counter[k])])
    return np.array(result)


# models

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# training

def learning_rate_decay(optimizer, lr_decay_ratio=0.9, min_lr=0.00001):
    for param_group in optimizer.param_groups:
        param_group['lr'] = max(param_group['lr'] * lr_decay_ratio)
    return optimizer

# logging

def ensure_path_exist(path):
    """
    Make path if it has not been made yet.
    """
    os.makedirs(path, exist_ok=True)
                    

def print_status(logger, epoch, epoch_mins, epoch_secs, train_loss, valid_loss):
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    logger.write(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    logger.write(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    logger.write(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')