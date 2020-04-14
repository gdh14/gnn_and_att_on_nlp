import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from utils.word_embedding import embedding
from torch.nn.utils import weight_norm

# class LSTM_clf(nn.Module):
    
#   def __init__(self, embed_dim, hidden_dim, vocab_size, out_size, 
#                layers=1, bidirectional=False):
#     super(LSTM_clf, self).__init__()
#     self.word_embedding = nn.Embedding(vocab_size, embed_dim)
#     self.net = nn.LSTM(embed_dim, hidden_dim,  num_layers=layers, 
#                        bidirectional=bidirectional, dropout=0.5)
#     self.relu = nn.ReLU()
#     self.bn = nn.BatchNorm1d(hidden_dim * (int(bidirectional) + 1))
#     self.linear = nn.Linear(hidden_dim * (int(bidirectional) + 1), out_size)

#   def forward(self, x):
#     out = self.word_embedding(x)
#     out = self.net(out)[0]
#     out = self.relu(out).transpose(1,2)
#     out = F.max_pool1d(out, out.size()[2]).squeeze()
#     out = self.linear(self.bn(out))
#     return out

class DCNN_block(nn.Module):
  
  def __init__(self, embed_dim, hidden_dim, kernel_size, dilations=None,
               dropout=0.2):
    super(DCNN_block, self).__init__()
    self.conv1 = weight_norm(nn.Conv1d(embed_dim, hidden_dim, kernel_size, dilation=1))
    self.conv2 = weight_norm(nn.Conv1d(embed_dim, hidden_dim, kernel_size, dilation=2))
    self.conv3 = weight_norm(nn.Conv1d(embed_dim, hidden_dim, kernel_size, dilation=4))
    self.net = nn.Sequential(self.conv1, nn.ReLU(), nn.Dropout(dropout),
                             self.conv2, nn.ReLU(), nn.Dropout(dropout), 
                             self.conv3, nn.ReLU(), nn.Dropout(dropout))
  
  def forward(self, x):
    # N x C x L
    return self.net(x)

class DCNN_rez_block(nn.Module):
  
  def __init__(self, embed_dim, hidden_dim, kernel_size, dilations=None,
               dropout=0.2):
    super(DCNN_rez_block, self).__init__()
    self.conv1 = weight_norm(nn.Conv1d(embed_dim, hidden_dim, kernel_size, 
                                       padding=(kernel_size-1)*1, dilation=1))
    self.conv2 = weight_norm(nn.Conv1d(embed_dim, hidden_dim, kernel_size, 
                                       padding=(kernel_size-1)*2, dilation=2))
    self.conv3 = weight_norm(nn.Conv1d(embed_dim, hidden_dim, kernel_size, 
                                       padding=(kernel_size-1)*4, dilation=4))

    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.relu3 = nn.ReLU()

    self.do1 = nn.Dropout(dropout)
    self.do2 = nn.Dropout(dropout)
    self.do3 = nn.Dropout(dropout)
  
  def forward(self, x):
    # N x C x L
    seq_len = x.size()[2]
    out = self.do1(self.relu1(self.conv1(x)))[:, :, -seq_len:]
    out = out + self.do2(self.relu2(self.conv2(x)))[:, :, -seq_len:]
    out = out + self.do3(self.relu3(self.conv3(x)))[:, :, -seq_len:]
    return out


class DCNN(nn.Module):

  def __init__(self, embed_dim, hidden_dim, vocab_size, out_size, 
               kernel_size, dilations=None, rez_block=True, 
               dropout=0.2):
    super(DCNN, self).__init__()
    self.word_embedding = nn.Embedding(vocab_size, embed_dim)
    if rez_block: 
      self.net = DCNN_rez_block(embed_dim, hidden_dim, kernel_size, dilations, dropout)
    else:
      self.net = DCNN_block(embed_dim, hidden_dim, kernel_size, dilations, dropout)
    self.bn = nn.BatchNorm1d(hidden_dim)
    self.do = nn.Dropout(dropout)
    self.linear = nn.Linear(hidden_dim, out_size)

  def forward(self, x):
    out = self.word_embedding(x)
    out = self.net(out.transpose(1,2))
    out = F.max_pool1d(out, out.size()[2]).squeeze()
    out = self.linear(self.do(self.bn(out)))
    return out


class DDCNN(nn.Module):
  # Dilated and Dense CNN
  def __init__(self, embed_dim, hidden_dim, vocab_size, out_size, 
               kernel_size, dilations=None, rez_block=True, 
               dropout=0.2):
    super(DDCNN, self).__init__()
    self.word_embedding = nn.Embedding(vocab_size, embed_dim)
    if rez_block: 
      self.dcnn = DCNN_rez_block(embed_dim, hidden_dim, kernel_size, dilations, dropout)
    else:
      self.dcnn = DCNN_block(embed_dim, hidden_dim, kernel_size, dilations, dropout)

    self.do1 = nn.Dropout(dropout)
    self.do2 = nn.Dropout(dropout)
    self.do3 = nn.Dropout(dropout)
    self.cnn1 = weight_norm(nn.Conv1d(embed_dim, int(hidden_dim//3), 4, padding=3, dilation=1))
    self.cnn2 = weight_norm(nn.Conv1d(embed_dim, int(hidden_dim//3), 6, padding=5, dilation=1))
    self.cnn3 = weight_norm(nn.Conv1d(embed_dim, int(hidden_dim//3), 8, padding=7, dilation=1))
    
    self.bn = nn.BatchNorm1d(hidden_dim*2)
    self.do = nn.Dropout(dropout)
    self.linear = nn.Linear(hidden_dim*2, out_size)

  def cnn(self, x):
    out1 = F.relu(self.cnn1(self.do1(x)))
    out2 = F.relu(self.cnn2(self.do2(x)))
    out3 = F.relu(self.cnn3(self.do3(x)))
    outs = []
    for o in [out1, out2, out3]:
      outs.append(F.max_pool1d(o, o.size()[2]).squeeze())
    out = torch.cat(outs, 1)
    return out

  def forward(self, x):
    out = self.word_embedding(x).transpose(1,2)
    dcnn_out = self.dcnn(out)
    cnn_out = self.cnn(out)
    dcnn_out = F.max_pool1d(dcnn_out, dcnn_out.size()[2]).squeeze()
    out = self.linear(self.do(self.bn(torch.cat((dcnn_out,cnn_out), 1))))
    return out



class MLP_block(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, batch_norm=False, dropout=1, out_layer=False):
        super(MLP_block, self).__init__()
        if batch_norm:
            self.bn = nn.BatchNorm1d(in_dim)
        self.linear = nn.Linear(in_dim, out_dim, bias)
        self.out_layer = out_layer
        if not out_layer:
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x):
        # x has input size: batch x length x features
        out = self.bn(x) if hasattr(self, 'bn') else x        
        out = self.linear(out)
        out = self.dropout(self.relu(out)) if not self.out_layer else out
        return out

# cite: https://github.com/pytorch/examples/blob/master/word_language_model/model.py    

# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, nout, ninp, nhead, nhid, nlayers, vocab_size, dropout=0.5, 
                 pretrained_embedding_path=None):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
#         self.embed = embedding(pretrained_embedding_path, vocab, ninp)
        self.embed = nn.Embedding(vocab_size, ninp)
        self.ninp = ninp
        self.decoder = MLP_block(ninp, nout, dropout=dropout, batch_norm=False, out_layer=True)
        
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, has_mask=False):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.embed(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
#         import pdb; pdb.set_trace()
        output = self.transformer_encoder(src).transpose(1,2)
#         output, _ = self.transformer_encoder(src, self.src_mask).max(1)
#         output_pool = self.pool(output.transpose(2,1)).view(len(output), -1)
#         import pdb; pdb.set_trace()
#         output = torch.cat([output_pool, output[:, -1, :].view(len(output), -1)], dim=1)
        output = F.max_pool1d(output, output.size()[2]).squeeze()
        output = self.decoder(output)
        return output #F.log_softmax(output, dim=-1)
    
    
from torch.nn.utils import weight_norm

class LSTM_clf(nn.Module):

    def __init__(self, embed_dim, hidden_dim, vocab_size, out_size, 
               layers=1, bidirectional=False):
        super(LSTM_clf, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.net = nn.LSTM(embed_dim, hidden_dim,  num_layers=layers, 
                           bidirectional=bidirectional, dropout=0.5)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(hidden_dim * (int(bidirectional) + 1))
        self.linear = nn.Linear(hidden_dim * (int(bidirectional) + 1), out_size)

    def forward(self, x):
        out = self.word_embedding(x)
#         import pdb; pdb.set_trace()
        out = self.net(out)[0]
        out = self.relu(out).transpose(1,2)
        out = F.max_pool1d(out, out.size()[2]).squeeze()
        out = self.linear(self.bn(out))
        return out
    
    
    