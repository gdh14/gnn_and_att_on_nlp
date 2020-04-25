import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import time

# reference: https://github.com/bentrevett/pytorch-seq2seq
class Attention(nn.Module):
    def __init__(self, enc_nhid, dec_nhid):
        super(Attention, self).__init__()
        self.attn = nn.Linear(enc_nhid*2+dec_nhid, dec_nhid)
        self.v = nn.Linear(dec_nhid, 1, bias=False)
        
    def forward(self, hidden, enc_out):
        # hidden: (b, nhid)
        # enc_out: (s, b, nhid*2)
        b = enc_out.shape[1]
        src_len = enc_out.shape[0]
        hidden = hidden[-1].unsqueeze(1).repeat(1, src_len, 1)
        enc_out = enc_out.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, enc_out), dim=2)))
        # energy: (b, s, nhid)
        attn = self.v(energy).squeeze(2)
        return F.softmax(attn, dim=1)
    
    
class GRUEncoder(nn.Module):
    def __init__(self, ninp, nembed, enc_nhid, dec_nhid, nlayers, dropout=0.2):
        super(GRUEncoder, self).__init__()
        self.embedding = nn.Embedding(ninp, nembed)
        self.nlayers = nlayers
        self.rnn = nn.GRU(nembed, enc_nhid, nlayers, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(enc_nhid*2, dec_nhid)
        
    def forward(self, src):
        # src: (s, b)
        s, b = src.shape
        src = self.dropout(self.embedding(src))
        # src: (s, b, nembed), hidden: (nlayers*dir, b, nhid), cell: (nlayers*dir, b, nhid)
        out, hidden = self.rnn(src)
        hidden = hidden.transpose(1,0).reshape(b, self.nlayers, -1).transpose(1,0)
        hidden = torch.tanh(self.fc(hidden))
        return out, hidden
    
    
class GRUDecoder(nn.Module):
    def __init__(self, nout, nembed, enc_nhid, dec_nhid, nlayers, dropout=0.2, attention=None):
        super(GRUDecoder, self).__init__()
        self.nout = nout
        self.nlayers = nlayers
        self.embedding = nn.Embedding(nout, nembed)
        self.rnn = nn.GRU(nembed+enc_nhid*2, dec_nhid, nlayers, 
                          dropout=dropout, bidirectional=False)
        self.fc_out = nn.Linear(enc_nhid*2+dec_nhid+nembed, nout)
        self.dropout = nn.Dropout(dropout)
        self.attention = attention
    
    def forward(self, src, hidden, enc_out):
        """
        for decoder, we process one token at a time
        """
        # src: (b, seq_len)
        embedded = src.unsqueeze(0)
        embedded = self.dropout(self.embedding(embedded))
        attn = self.attention(hidden, enc_out)
        attn = attn.unsqueeze(1)
        enc_out = enc_out.permute(1,0,2)
        ctxt = torch.bmm(attn, enc_out).permute(1,0,2)
        x = torch.cat((embedded, ctxt), dim=2)
        out, hidden = self.rnn(x, hidden)
        # out, hidden: (1, b, nhid)
        
        embedded = embedded.squeeze(0)
        out = out.squeeze(0)
        ctxt = ctxt.squeeze(0)
        pred = self.fc_out(torch.cat((out, ctxt, embedded), dim=1))
        return pred, hidden, attn
        
        
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
        attns = []
        for t in range(1, tgt_len):
            out, hidden, attn = self.decoder(x, hidden, enc_out)
            attns.append(attn)
            outs[t] = out
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = out.argmax(1)
            x = tgt[t] if teacher_force else top1
        attns = torch.stack(attns, dim=1).squeeze() # (b, tgt_len, src_len)
        return outs, attns
    

def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50, src_language='de'):
    model.eval()
    if isinstance(sentence, str):
        nlp = spacy.load('de')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    src_len = torch.LongTensor([len(src_indexes)]).to(device)
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)
    mask = model.create_mask(src_tensor)
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)
    
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)
        attentions[i] = attention
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:], attentions[:len(trg_tokens)-1]


def display_attention(sentence, translation, attention, cmap='coolwarm'):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    attention = attention.squeeze(1).cpu().detach().numpy()
    cax = ax.matshow(attention, cmap=cmap)
    ax.tick_params(labelsize=15)
    ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'], 
                       rotation=45)
    ax.set_yticklabels(['']+translation)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()
    

def bleu_and_attention(data, src_field, trg_field, model, device, max_len = 50):
    from torchtext.data.metrics import bleu_score
    trgs = []
    pred_trgs = []
    attns = []
    
    for datum in data:
        src = vars(datum)['src']
        trg = vars(datum)['trg']
        pred_trg, attn = translate_sentence(src, src_field, trg_field, model, device, max_len)
        #cut off <eos> token
        pred_trg = pred_trg[:-1]
        pred_trgs.append(pred_trg)
        attns.append(attn)
        trgs.append([trg])
        
    return bleu_score(pred_trgs, trgs), attns