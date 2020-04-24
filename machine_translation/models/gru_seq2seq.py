#!/usr/bin/env python
import torch, spacy, random, math, time
import torch.nn as nn


class GRUEncoder(nn.Module):
    def __init__(self, ninp, nembed, enc_nhid, dec_nhid, nlayers, dropout):
        super(GRUEncoder, self).__init__()
        self.enc_nhid = enc_nhid
        self.dec_nhid = dec_nhid
        self.nlayers = nlayers
        self.embedding = nn.Embedding(ninp, nembed)
        self.rnn = nn.GRU(nembed, enc_nhid, nlayers, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(enc_nhid*2, dec_nhid)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src: (src len, b)
        s, b = src.shape
        embedded = self.dropout(self.embedding(src))
        out, hidden = self.rnn(embedded)
        hidden = hidden.transpose(1,0).reshape(b, self.nlayers, -1).transpose(1,0)
        hidden = torch.tanh(self.fc(hidden))
        return out, hidden
    
    
class GRUDecoder(nn.Module):
    def __init__(self, nout, nembed, enc_nhid, dec_nhid, nlayers, dropout):
        super(GRUDecoder, self).__init__()
        self.nout = nout
        self.enc_nhid = enc_nhid
        self.dec_nhid = dec_nhid
        self.nlayers = nlayers
        self.embedding = nn.Embedding(nout, nembed)
        self.rnn = nn.GRU(nembed, dec_nhid, nlayers, bidirectional=False, dropout=dropout)
        self.fc_out = nn.Linear(dec_nhid, nout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden):
        x = x.unsqueeze(0)
        embedded = self.dropout(self.embedding(x))
        out, hidden = self.rnn(embedded, hidden)
        pred = self.fc_out(out.squeeze(0))
        return pred, hidden
    

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # src: (src_len, b)
        # tgt: (tgt_len, b)
        tgt_len, b = tgt.shape
        tgt_vocab_size = self.decoder.nout
        
        # tensor to store decoder outputs
        outs = torch.zeros(tgt_len, b, tgt_vocab_size).to(self.device)
        
        enc_out, hidden = self.encoder(src)
        x = tgt[0]
        for t in range(1, tgt_len):
            out, hidden = self.decoder(x, hidden)
            outs[t] = out
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = out.argmax(1)
            x = tgt[t] if teacher_force else top1
            
        return outs

    
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TGT.vocab)
    ENC_EMB_DIM = 250
    DEC_EMB_DIM = 250
    ENC_HID_DIM = 500
    DEC_HID_DIM = 500
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    NLAYERS = 2

    enc = GRUEncoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, NLAYERS, ENC_DROPOUT)
    dec = GRUDecoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, NLAYERS, DEC_DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)
    print(f'The model has {count_parameters(model):,} trainable parameters')
