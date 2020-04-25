import torch
import torch.nn as nn
import random
import math
import time

# reference: https://github.com/bentrevett/pytorch-seq2seq
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, nhid, nheads, dropout, device):
        super(MultiHeadAttentionLayer, self).__init__()
        assert(nhid % nheads == 0)
        
        self.nhid = nhid
        self.nheads = nheads
        self.head_dim = nhid // nheads
        
        self.fc_q = nn.Linear(nhid, nhid)
        self.fc_k = nn.Linear(nhid, nhid)
        self.fc_v = nn.Linear(nhid, nhid)
        self.fc_o = nn.Linear(nhid, nhid)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)


    def forward(self, query, key, value, mask=None):
        b = query.shape[0]
        
        # query: (b, ql, nhid), key: (b, kl, nhid), value: (b, vl, nhid)
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        Q = Q.view(b, -1, self.nheads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(b, -1, self.nheads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(b, -1, self.nheads, self.head_dim).permute(0, 2, 1, 3)
        # Q: (b, nheads, ql, head_dim)
        # K: (b, nheads, kl, head_dim)
        # V: (b, nheads, vl, head_dim)
        
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy: (b, nheads, ql, kl)
        if mask is not None:
            energy = energy.masked_fill(mask==0, -1e10)
        
        attn = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attn), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(b, -1, self.nhid)
        x = self.fc_o(x)
        return x, attn


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, nhid, pfdim, dropout):
        super(PositionwiseFeedforwardLayer, self).__init__()
        self.fc_1 = nn.Linear(nhid, pfdim)
        self.fc_2 = nn.Linear(pfdim, nhid)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (b, s, nhid)
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x

    
class Encoder(nn.Module):
    def __init__(self, ninp, nhid, nlayers, nheads, pfdim, dropout, device, max_len=100):
        super(Encoder, self).__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(ninp, nhid)
        self.pos_embedding = nn.Embedding(max_len, nhid)
        self.layers = nn.ModuleList([EncoderLayer(nhid, nheads, pfdim, dropout, device)
                                     for _ in range(nlayers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([nhid])).to(device)
        
    def forward(self, src, src_mask):
        # src: (b, s)
        # src_mask: (b, s)
        b, s = src.shape
        pos = torch.arange(0, s).unsqueeze(0).repeat(b, 1).to(self.device)
        src = self.dropout((self.tok_embedding(src)*self.scale+self.pos_embedding(pos)))
        for layer in self.layers:
            src = layer(src, src_mask)
        return src


class EncoderLayer(nn.Module):
    def __init__(self, nhid, nheads, pfdim, dropout, device, max_len=100):
        super(EncoderLayer, self).__init__()
        self.self_attn_layer_norm = nn.LayerNorm(nhid)
        self.ff_layer_norm = nn.LayerNorm(nhid)
        self.self_attn = MultiHeadAttentionLayer(nhid, nheads, dropout, device)
        self.positionwise_ff = PositionwiseFeedforwardLayer(nhid, pfdim, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        # src: (b, s, nhid)
        # src_mask: (b, s)
        _src, _ = self.self_attn(src, src, src, src_mask)
        src = self.self_attn_layer_norm(src+self.dropout(_src))
        # src: (b, s, nhid)
        _src = self.positionwise_ff(src)
        src = self.ff_layer_norm(src+self.dropout(_src))
        # src: (b, s, nhid)
        return src
    
    
class Decoder(nn.Module):
    def __init__(self, nout, nhid, nlayers, nheads, pfdim, dropout, device, max_len=100):
        super(Decoder, self).__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(nout, nhid)
        self.pos_embedding = nn.Embedding(max_len, nhid)
        self.layers = nn.ModuleList([DecoderLayer(nhid, nheads, pfdim, dropout, device)
                                     for _ in range(nlayers)])
        self.fc_out = nn.Linear(nhid, nout)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([nhid])).to(device)
        
    def forward(self, tgt, enc_src, tgt_mask, src_mask):
        b, tgt_len = tgt.shape
        pos = torch.arange(0, tgt_len).unsqueeze(0).repeat(b, 1).to(self.device)
        tgt = self.dropout((self.tok_embedding(tgt)*self.scale)+self.pos_embedding(pos))
        for layer in self.layers:
            tgt, attn = layer(tgt, enc_src, tgt_mask, src_mask)
        out = self.fc_out(tgt)
        return out, attn


class DecoderLayer(nn.Module):
    def __init__(self, nhid, nheads, pfdim, dropout, device):
        super(DecoderLayer, self).__init__()
        self.self_attn_layer_norm = nn.LayerNorm(nhid)
        self.enc_attn_layer_norm = nn.LayerNorm(nhid)
        self.ff_layer_norm = nn.LayerNorm(nhid)
        self.self_attn = MultiHeadAttentionLayer(nhid, nheads, dropout, device)
        self.encoder_attn = MultiHeadAttentionLayer(nhid, nheads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(nhid, pfdim, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, enc_src, tgt_mask, src_mask):
        """
        tgt: (b, tgt_len, nhid)
        enc_src: (b, sec_len, nhid)
        tgt_mask: (b, tgt_len)
        src_mask: (b, src_len)
        """
        _tgt, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.self_attn_layer_norm(tgt+self.dropout(_tgt))
        
        _tgt, attn = self.encoder_attn(tgt, enc_src, enc_src, src_mask)
        tgt = self.enc_attn_layer_norm(tgt+self.dropout(_tgt))
        
        _tgt = self.positionwise_feedforward(tgt)
        tgt = self.ff_layer_norm(tgt+self.dropout(_tgt))
        return tgt, attn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, tgt_pad_idx, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        # src: (b, s)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask: (b, 1, 1, s)
        return src_mask
    
    def make_tgt_mask(self, tgt):
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        # tgt_pad_mask: (b, 1, 1, tgt len)
        tgt_len = tgt.shape[1]
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=self.device)).bool()
        # tgt_sub_mask: (b, tgt len)
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        # tgt_mask: (b, 1, tgt len, tgt len)
        
    def forward(self, src, tgt):
        # src: (b, src len)
        # tgt: (b, tgt len)
        src_mask = self.make_src_mask(src)  # (b, 1, 1, src len)
        tgt_mask = self.make_tgt_mask(tgt)  # (b, 1, tgt len, tgt len)
        
        enc_src = self.encoder(src, src_mask)
        out, attn = self.decoder(tgt, enc_src, tgt_mask, src_mask)
        # out: (b, tgt len, nout)
        # attn: (b, nheads, tgt len, src len)
        return out, attn
    
    
# analysis and metrics

def translate_sentence(sentence, src_field, tgt_field, model, device, max_len=100, src_language='de'):
    model.eval()
    if isinstance(sentence, str):
        import spacy
        nlp = spacy.load(src_language)
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    src_idxs = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_idxs).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
    tgt_idxs = [tgt_field.vocab.stoi[tgt_field.init_token]]
    
    for i in range(max_len):
        tgt_tensor = torch.LongTensor(tgt_idxs).unsqueeze(0).to(device)
        tgt_mask = model.make_tgt_mask(tgt_tensor)
        with torch.no_grad():
            out, attn = model.decoder(tgt_tensor, enc_src, tgt_mask, src_mask)
        pred_token = out.argmax(2)[:,-1].item()
        tgt_idxs.append(pred_token)
        
        if pred_token == tgt_field.vocab.stoi[tgt_field.eos_token]:
            break
            
    tgt_tokens = [tgt_field.vocab.itos[i] for i in tgt_idxs]
    return tgt_tokens[1:], attn
    
    
def display_attention(sentence, translation, attention, n_head=8, n_rows=4, n_cols=2, cmap='coolwarm'):
    assert(n_rows * n_cols == n_head)
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(15,25))
    for i in range(n_heads):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        attn = attention.squeeze(0)[i].cpu().detach().numpy()
        cax = ax.matshow(attn, cmap=cmap)
        
        ax.tick_params(labelsize=12)
        ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'], 
                           rotation=45)
        ax.set_yticklabels(['']+translation)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()
        

def bleu_and_attention(data, src_field, tgt_field, model, device, max_len=100):
    from torchtext.data.metrics import bleu_score
    tgts = []
    pred_tgts = []
    attns = []
    for d in data:
        src = vars(d)['src']
        tgt = vars(d)['trg']
        pred_tgt, attn = translate_sentence(src, src_field, tgt_field, model, device, max_len)
        pred_tgt = pred_tgt[:-1]
        pred_tgts.append(pred_tgt)
        attns.append(attn)
        tgts.append([tgt])
    return bleu_score(pred_tgts, tgts), attns