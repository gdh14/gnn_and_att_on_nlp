import sys

sys.path.append("..")

import yaml, json
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
from utils.early_stopping import EarlyStopping


####################
#     Functions
####################

def accuracy(preds, y):
    return (np.array(preds) == np.array(y)).astype(int).mean()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch(epoch, model, optimizer, criterion, early_stopper=None):
    model.train()
    train_loss, n_data = 0, 0
    start = time.time()
    preds = []
    labels = []
    counter = 0
    for i, (x, y, seq_lens) in enumerate(train_loader):
        if x.shape[1] > args["max_doc_len"]:
            print(x.shape)
        n_data += x.size()[0]
        labels.extend(y.cpu().detach().tolist())
        if torch.cuda.is_available(): x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        out, att = model(x, seq_lens)
        preds.extend(out.argmax(axis=1).cpu().detach().tolist())
        loss = criterion(out, y)
        loss.backward()
        if grad_clip: torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        train_loss += loss
        counter += 1
        
        if i % print_iter == print_iter - 1:
            
            model, valid_preds, valid_labels, valid_loss = validate(model, nn.CrossEntropyLoss(), valid_loader)
            print("""epoch {} - batch [{}/{}] - train loss: {:.2f} - acc: {:.3f} - valid loss : {:.2f} - acc : {:.3f} time taken: {:.2f}""".format(epoch, i, 
                len(train_loader), train_loss/counter,#(i+1),
                accuracy(preds, labels), valid_loss, accuracy(valid_preds, valid_labels),
                time.time()-start), flush=True)

            model.train()
            if early_stopper is not None:
                early_stopper(valid_loss, model)
            start = time.time()
            train_loss, counter = 0, 0
                
    model, valid_preds, valid_labels, valid_loss = validate(model, nn.CrossEntropyLoss(), valid_loader)
    print("""epoch {} - batch [{}/{}] - train loss: {:.2f} - acc: {:.3f} - valid loss : {:.2f} - acc : {:.3f} time taken: {:.2f}""".format(epoch, i, 
        len(train_loader), train_loss/(i+1),
        accuracy(preds, labels), valid_loss, accuracy(valid_preds, valid_labels),
        time.time()-start), flush=True)
    if early_stopper is not None:
        early_stopper(valid_loss, model)
    return model

def learning_rate_decay(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = max(param_group['lr'] * lr_decay_ratio, 0.00001)
    return optimizer

def training(model, epoches, lr, wd):
    if torch.cuda.is_available():
        model.cuda()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopping(model_dir, patience=PATIENCE)
    for ep in range(epoches):
        model = train_epoch(ep, model, optimizer, criterion, early_stopper)
        optimizer = learning_rate_decay(optimizer)
        
        if early_stopper.early_stop:
            return model
    return model

def validate(model, criterion, loader):
    model.eval()
    valid_loss = 0
    preds, labels = [], []
    for i, (x, y, seq_lens) in enumerate(loader):
        labels.extend(y.cpu().detach().tolist())
        if torch.cuda.is_available(): x, y = x.cuda(), y.cuda()
        out, att = model(x, seq_lens)
        loss = criterion(out, y)
        preds.extend(out.cpu().detach().argmax(axis=1).tolist())
        valid_loss += loss

    return model, preds, labels, valid_loss.item()/len(loader)
    
def predict(model, loader):
    model.eval()
    preds, labels = [], []
    for i, (x, _, seq_lens) in enumerate(loader):
        if torch.cuda.is_available(): x = x.cuda()
        out, att = model(x, seq_lens)
        preds.extend(out.cpu().detach().argmax(axis=1).tolist())
    return preds

def dict_kv2vk(d):
    result = {}
    for k, v in d.items():
        result[v] = k
    return result

def int_dict_increment_by(d, number):
    result = {}
    for k, v in d.items():
        result[k] = v+number
    return result

# data prep
torch.manual_seed(11747)
start = time.time()
config_file = sys.argv[1] #'config.yaml'  
config = yaml.load(open(config_file), Loader=yaml.FullLoader)
args = config["training"]
text_data = TextDataForTC(config_file, config['preprocess']['dataset'])
label2idx = text_data.get_label2idx()
vocab = text_data.get_vocab()
text_ls_train = text_data.get_text_ls('train')
text_ls_val = text_data.get_text_ls('val')
text_ls_test = text_data.get_text_ls('test')
label_ls_train = text_data.get_label_ls('train')
label_ls_val = text_data.get_label_ls('val')
label_ls_test = text_data.get_label_ls('test')

word2idx = vocab.get_word2idx()
word2idx = int_dict_increment_by(vocab.get_word2idx(),2)
word2idx["<pad>"] = 0
word2idx["<unk>"] = 1
idx2word = dict_kv2vk(word2idx)

print("dataloaded...time taken: " + str(time.time() - start))

train_x = text_2_int_list(text_ls_train, word2idx, args["max_doc_len"])
valid_x = text_2_int_list(text_ls_val, word2idx, args["max_doc_len"])
test_x  = text_2_int_list(text_ls_test, word2idx, args["max_doc_len"])

# datasets 
train = MyDataset(train_x, label_ls_train)
valid = MyDataset(valid_x, label_ls_val)
test = MyDataset(test_x, label_ls_test)

# hyper-parameters
task = config["preprocess"]["dataset"]
embed_dim = args["embed_dim"]
nhead = args["nhead"]
nhid = args["nhid"]
nlayers = args["nlayers"]
vocab_size = len(word2idx)
bs = args["batch_size"]
nclass = len(label2idx)
lr = args["lr"]
grad_clip = args["grad_clip"]
print_iter = args["print_iter"]
dropout = args["dropout"]
wd = args["weight_decay"]
epochs = args["epochs"]
model_id = args["id"]
kernel_size = 5
model_dir = os.path.join(args["model_dir"]+"_"+task, args["model"] + str(model_id))
is_cuda = torch.cuda.is_available()
device = 'cuda:0' if is_cuda else 'cpu'
PATIENCE = 5
bidir = True
context_size = nhid//2
lr_decay_ratio = args["lr_decay_ratio"]

args["nclass"] = nclass
args["vocab_size"] = vocab_size
args["kernel_size"] = kernel_size
print("vocab size: ", vocab_size)

# data loaders
train_loader = DataLoader(train, batch_size=bs, shuffle=True, collate_fn=batchify)
valid_loader = DataLoader(valid, batch_size=bs, shuffle=False, collate_fn=batchify)
test_loader = DataLoader(test, batch_size=bs, shuffle=False, collate_fn=batchify)

print("dataloader created...")

# model

if args["model"] == 'lstm':
    from models.classifiers import LSTM_clf
    model = LSTM_clf(embed_dim, nhid, vocab_size, 
                     nclass, nlayers, bidir, context_size, device=device)
    
elif args["model"] == 'transformer':
    from models.classifiers import TransformerModel
    model = TransformerModel(nclass, embed_dim, nhead, nhid, nlayers, 
                             len(word2idx), dropout, context_size, device=device)
    
elif args["model"] == "ddcnn":
    from models.classifiers import DDCNN
    model = DDCNN(embed_dim, nhid, len(word2idx), nclass, kernel_size, rez_block=True)
    
print("num of model params: ", count_parameters(model))
    
print("start training...")
training(model, epochs, lr, wd)

with open(os.path.join(model_dir, "args.json"), "w") as f:
    json.dump(args, f)

print("testing...")

# load the best performing model
state_dict = torch.load(os.path.join(model_dir, "checkpoint.pt"), map_location='cpu')
model.load_state_dict(state_dict["model_state_dict"])

if is_cuda:
    model.cuda()
            
_, valid_preds, valid_labels, valid_loss = validate(model, nn.CrossEntropyLoss(), valid_loader)
print("valid loss: {}, valid accuracy: {}".format(valid_loss, accuracy(valid_preds, valid_labels)))

    
_, test_preds, test_labels, test_loss = validate(model, nn.CrossEntropyLoss(), test_loader)
print("test loss: {}, test accuracy: {}".format(test_loss, accuracy(test_preds, test_labels)))

results = {"valid_loss": valid_loss, 
           "valid_acc": accuracy(valid_preds, valid_labels),
           "test_loss": test_loss,
           "test_acc": accuracy(test_preds, test_labels)}
    
with open(os.path.join(model_dir, "results.json"), "w") as f:
    json.dump(results, f)
