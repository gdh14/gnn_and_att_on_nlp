import torch
import torch.nn as nn
from tqdm import tqdm

################################################
#       training functions for seq encoders
################################################


def train_epoch_gru(model, iterator, optimizer, criterion, clip):
    model.train()
    train_loss = 0
    for i, batch in tqdm(enumerate(iterator)):
        src = batch.src
        tgt = batch.trg
        optimizer.zero_grad()
        out = model(src, tgt)
        out_dim = out.shape[-1]
        out = out[1:].view(-1, out_dim)
        tgt = tgt[1:].view(-1)
        
        loss = criterion(out, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        train_loss += loss.item()
    return train_loss/len(iterator)


def evaluate_gru(model, iterator, criterion):
    model.eval()
    loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            tgt = batch.trg
            out = model(src, tgt, 0) # no teacher forcing here
            out_dim = out.shape[-1]
            out = out[1:].view(-1, out_dim)
            tgt = tgt[1:].view(-1)
            loss += criterion(out, tgt).item()
    return loss/len(iterator)


def train_epoch_gru_attn(model, iterator, optimizer, criterion, clip):
    model.train()
    train_loss = 0
    for i, batch in tqdm(enumerate(iterator)):
        src = batch.src
        tgt = batch.trg
        optimizer.zero_grad()
        out, attns = model(src, tgt)
        out_dim = out.shape[-1]
        out = out[1:].view(-1, out_dim)
        tgt = tgt[1:].view(-1)
        
        loss = criterion(out, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        train_loss += loss.item()
    return train_loss/len(iterator)


def evaluate_gru_attn(model, iterator, criterion):
    model.eval()
    loss = 0
    attns = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            tgt = batch.trg
            out, attn = model(src, tgt, 0) # no teacher forcing here
            attns.append(attn)
            out_dim = out.shape[-1]
            out = out[1:].view(-1, out_dim)
            tgt = tgt[1:].view(-1)
            loss += criterion(out, tgt).item()
    return loss/len(iterator), attns


def train_epoch_tfmr(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in tqdm(enumerate(iterator)):
        src = batch.src.t()
        tgt = batch.trg.t()     
        optimizer.zero_grad()
        out, _ = model(src, tgt[:,:-1])  # tgt sent excluding <eos>
        nout = out.shape[-1]

        out = out.contiguous().view(-1, nout)
        tgt = tgt[:,1:].contiguous().view(-1)
        
        loss = criterion(out, tgt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def evaluate_tfmr(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    attns = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src.t()
            tgt = batch.trg.t()
            out, attn = model(src, tgt[:,:-1])
            attns.append(attn)
            nout = out.shape[-1]
            out = out.contiguous().view(-1, nout)
            tgt = tgt[:,1:].contiguous().view(-1)
            loss = criterion(out, tgt)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator), attns


def epoch_time(start, end):
    elapsed_time = start - end
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


################################################
#       training functions for gcn encoders
################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch_gcn_gru(model, iterator, optimizer, criterion, clip):
    model.train()
    train_loss = 0
    for i, batch in tqdm(enumerate(iterator)):
        src = batch.src.to(device)
        tgt = batch.trg.to(device)
        grh = batch.grh.to(device)
        optimizer.zero_grad()
        out = model(src, grh, tgt)
        out_dim = out.shape[-1]
        out = out[1:].view(-1, out_dim)
        tgt = tgt[1:].view(-1)
        
        loss = criterion(out, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        train_loss += loss.item()
    return train_loss/len(iterator)


def evaluate_gcn_gru(model, iterator, criterion):
    model.eval()
    loss = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(iterator)):
            src = batch.src.to(device)
            tgt = batch.trg.to(device)
            grh = batch.grh.to(device)
            out = model(src, grh, tgt, 0) # no teacher forcing here
            out_dim = out.shape[-1]
            out = out[1:].view(-1, out_dim)
            tgt = tgt[1:].view(-1)
            loss += criterion(out, tgt).item()
    return loss/len(iterator)