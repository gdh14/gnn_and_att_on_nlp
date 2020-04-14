import sys

sys.path.append("..")

import yaml
import numpy as np
import scipy.sparse as sp
import pickle as pkl
import os
import argparse

from preprocess.utils import DocumentStatsBuilder as DSBuilder
from preprocess.utils import TextConverter
from preprocess.data import TextDataForTC
import time

import torch
from torch import nn, LongTensor
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch.optim as optim
import torch.nn.functional as F
from IPython.core.debugger import set_trace
from dataloader import MyDataset, batchify, batchify_test, text_2_int_list



####################
#     Functions
####################

def accuracy(preds, y):
    return (np.array(preds) == np.array(y)).astype(int).mean()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch(epoch, model, optimizer, criterion):
    model.train()
    train_loss, n_data = 0, 0
    start = time.time()
    preds = []
    labels = []
    counter = 0
    for i, (x, y) in enumerate(train_loader):
        if x.shape[1] > 400:
            print(x.shape)
        n_data += x.size()[0]
        labels.extend(y.cpu().detach().tolist())
        if torch.cuda.is_available(): x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
#         import pdb; pdb.set_trace()
        out = model(x)
        preds.extend(out.argmax(axis=1).cpu().detach().tolist())
        loss = criterion(out, y)
        loss.backward()
        if grad_clip: torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        train_loss += loss
        counter += 1
        
        if i % print_iter == print_iter - 1:
            
            model, valid_preds, valid_labels, valid_loss = validate(model, criterion)
            print("""epoch {} - batch [{}/{}] - train loss: {:.2f} - acc: {:.3f} - valid loss : {:.2f} - acc : {:.3f} time taken: {:.2f}""".format(epoch, i, 
                len(train_loader), train_loss/counter,#(i+1),
                accuracy(preds, labels), valid_loss, accuracy(valid_preds, valid_labels),
                time.time()-start), flush=True)

        
            model.train()
            start = time.time()
            train_loss, counter = 0, 0
    
#     import pdb; pdb.set_trace()
    model, valid_preds, valid_labels, valid_loss = validate(model, criterion)
    print("""epoch {} - batch [{}/{}] - train loss: {:.2f} - acc: {:.3f} - valid loss : {:.2f} - acc : {:.3f} time taken: {:.2f}""".format(epoch, i, 
        len(train_loader), train_loss/(i+1),
        accuracy(preds, labels), valid_loss, accuracy(valid_preds, valid_labels),
        time.time()-start), flush=True)
    return model

def learning_rate_decay(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.9
    return optimizer

def training(model, epoches, lr, wd):
    if torch.cuda.is_available():
        model.cuda()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    for ep in range(epoches):
        model = train_epoch(ep, model, optimizer, criterion)
        optimizer = learning_rate_decay(optimizer)
    return model

def validate(model, criterion):
    model.eval()
    valid_loss = 0
    preds, labels = [], []
    for i, (x, y) in enumerate(valid_loader):
        labels.extend(y.cpu().detach().tolist())
        if torch.cuda.is_available(): x, y = x.cuda(), y.cuda()
        out = model(x)
        loss = criterion(out, y)
        preds.extend(out.cpu().detach().argmax(axis=1).tolist())
        valid_loss += loss
#     print("validating")
#     import pdb; pdb.set_trace()
    return model, preds, labels, valid_loss/(i+1)
    
def predict(model, loader):
    model.eval()
    preds, labels = [], []
    for i, (x, _) in enumerate(loader):
        if torch.cuda.is_available(): x = x.cuda()
        out = model(x)
        preds.extend(out.cpu().detach().argmax(axis=1).tolist())
    return preds


# data prep
start = time.time()
config_file = sys.argv[1] #'config.yaml'  
config = yaml.load(open(config_file), Loader=yaml.FullLoader)
args = config["training"]
text_data = TextDataForTC(config_file, config['preprocess']['dataset'])
label2idx = text_data.get_label2idx()
vocab = text_data.get_vocab()
word2idx = vocab.get_word2idx()
text_ls_train = text_data.get_text_ls('train')
text_ls_val = text_data.get_text_ls('val')
text_ls_test = text_data.get_text_ls('test')
label_ls_train = text_data.get_label_ls('train')
label_ls_val = text_data.get_label_ls('val')
label_ls_test = text_data.get_label_ls('test')

w2i = vocab.get_word2idx()
w2i["<unk>"] = len(w2i)

print("dataloaded...time taken: " + str(time.time() - start))

train_x = text_2_int_list(text_ls_train, w2i, args["max_doc_len"])
valid_x = text_2_int_list(text_ls_val, w2i, args["max_doc_len"])
test_x  = text_2_int_list(text_ls_test, w2i, args["max_doc_len"])

# datasets 
train = MyDataset(train_x, label_ls_train)
valid = MyDataset(valid_x, label_ls_val)
test = MyDataset(test_x)

# hyper-parameters
embed_dim = args["embed_dim"]
nhead = args["nhead"]
nhid = args["nhid"]
nlayers = args["nlayers"]
vocab_size = len(w2i)
bs = args["batch_size"]
nclass = len(label2idx)
lr = args["lr"]
grad_clip = args["grad_clip"]
print_iter = args["print_iter"]
dropout = args["dropout"]
wd = args["weight_decay"]
epochs = args["epochs"]

# data loaders
train_loader = DataLoader(train, batch_size=bs, shuffle=True, collate_fn=batchify)
valid_loader = DataLoader(valid, batch_size=bs, shuffle=False, collate_fn=batchify)
test_loader = DataLoader(test, batch_size=bs, shuffle=False, collate_fn=batchify_test)

print("dataloader created...")

# model

if args["model"] == 'lstm':
    from models.classifiers import LSTM_clf
    model = LSTM_clf(embed_dim, nhid, vocab_size, nclass, nlayers)
    
elif args["model"] == 'transformer':
    from models.classifiers import TransformerModel
    model = TransformerModel(nclass, embed_dim, nhead, nhid, nlayers, len(w2i), dropout)
    
elif args["model"] == "ddcnn":
    from models.classifiers import DDCNN
    model = DDCNN(embed_dim, nhid, len(w2i), nclass, 4, rez_block=True)
    

print("num of model params: ", count_parameters(model))
    
print("start training...")
training(model, epochs, lr, wd)