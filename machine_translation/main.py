import spacy, random, math, time, yaml, sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext.datasets import TranslationDataset, Multi30k, IWSLT
from torchtext.data import Field, BucketIterator, RawField, Dataset

from models.gcn import GCNLayer
from src.utils import set_seed, tokenize_de, tokenize_en, batch_graph, \
initialize_weights, get_sentence_lengths, counter2array, ensure_path_exist, \
print_status, learning_rate_decay
from src.early_stopping import EarlyStopping
from src.logging import Logger


# parameters
config_file = sys.argv[1] # 'config.yaml'
config = yaml.load(open(config_file), Loader=yaml.FullLoader)
args = config["training"]
SEED = args["seed"]
DATASET = args["dataset"]  # Multi30k or ISWLT
MODEL = args["model"]  # gru**2, gru_attn**2, transformer, gcn_gru
REVERSE = args["reverse"]
BATCH_SIZE = args["batch_size"]
ENC_EMB_DIM = args["encoder_embed_dim"]
DEC_EMB_DIM = args["decoder_embed_dim"]
ENC_HID_DIM = args["encoder_hidden_dim"]
DEC_HID_DIM = args["decoder_hidden_dim"]
ENC_DROPOUT = args["encoder_dropout"]
DEC_DROPOUT = args["decoder_dropout"]
NLAYERS = args["num_layers"]
N_EPOCHS = args["num_epochs"]
CLIP = args["grad_clip"]
LR = args["lr"]
LR_DECAY_RATIO = args["lr_decay_ratio"]
ID = args["id"]
PATIENCE = args["patience"]
DIR = 'checkpoints/{}-{}-{}/'.format(DATASET, MODEL, ID)
MODEL_PATH = DIR
LOG_PATH = '{}log.log'.format(DIR)
CONFIG_PATH = '{}config.yaml'.format(DIR)
set_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ensure_path_exist(DIR)
yaml.dump(config, open(CONFIG_PATH, 'w'))
    
if 'transformer' in MODEL:
    ENC_HEADS = args["encoder_heads"]
    DEC_HEADS = args["decoder_heads"]
    ENC_PF_DIM = args["encoder_pf_dim"]
    DEC_PF_DIM = args["decoder_pf_dim"]
    MAX_LEN = args["max_len"]
    
# dataset

SRC = Field(tokenize = lambda text: tokenize_de(text, REVERSE), 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)
TGT = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)
GRH = RawField(postprocessing=batch_graph)
data_fields = [('src', SRC), ('trg', TGT), ('grh', GRH)]

train_data = Dataset(torch.load("data/Multi30k/train_data.pt"), data_fields)
valid_data = Dataset(torch.load("data/Multi30k/valid_data.pt"), data_fields)
test_data = Dataset(torch.load("data/Multi30k/test_data.pt"), data_fields)

# dataloader
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE, 
    sort_key = lambda x: len(x.src),
    sort_within_batch=False,
    device = device)

# build vocab and print basic stats
SRC.build_vocab(train_data, min_freq = 2)
TGT.build_vocab(train_data, min_freq = 2)

print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of validation examples: {len(valid_data.examples)}")
print(f"Number of testing examples: {len(test_data.examples)}")
print(f"Unique tokens in source (de) vocabulary: {len(SRC.vocab)}")
print(f"Unique tokens in target (en) vocabulary: {len(TGT.vocab)}")

src_c, tgt_c = get_sentence_lengths(train_data)
src_lengths = counter2array(src_c)
tgt_lengths = counter2array(tgt_c)

print("maximum src, tgt sent lengths: ")
np.quantile(src_lengths, 1), np.quantile(tgt_lengths, 1)

# Get models and corresponding training scripts

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TGT.vocab)
SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TGT_PAD_IDX = TGT.vocab.stoi[TGT.pad_token]
    
if MODEL == "gru**2":  # gru**2, gru_attn**2, transformer, gcn_gru
    from models.gru_seq2seq import GRUEncoder, GRUDecoder, Seq2Seq
    enc = GRUEncoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, NLAYERS, ENC_DROPOUT)
    dec = GRUDecoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, NLAYERS, DEC_DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)
    
    from src.train import train_epoch_gru, evaluate_gru, epoch_time
    train_epoch = train_epoch_gru
    evaluate = evaluate_gru
    
elif MODEL == "gru_attn**2":
    from models.gru_attn import GRUEncoder, GRUDecoder, Seq2Seq, Attention
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = GRUEncoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, NLAYERS, ENC_DROPOUT)
    dec = GRUDecoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, NLAYERS, DEC_DROPOUT, attn)
    model = Seq2Seq(enc, dec, device).to(device)
    
    from src.train import train_epoch_gru, evaluate_gru, epoch_time
    train_epoch = train_epoch_gru
    evaluate = evaluate_gru
    
elif MODEL == "transformer":
    from models.transformer import Encoder, Decoder, Seq2Seq
    enc = Encoder(INPUT_DIM, ENC_HID_DIM, NLAYERS, ENC_HEADS, 
                  ENC_PF_DIM, ENC_DROPOUT, device, MAX_LEN)
    dec = Decoder(OUTPUT_DIM, DEC_HID_DIM, NLAYERS, DEC_HEADS, 
                  DEC_PF_DIM, DEC_DROPOUT, device, MAX_LEN)
    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TGT_PAD_IDX, device).to(device)
    
    from src.train import train_epoch_tfmr, evaluate_tfmr, epoch_time
    train_epoch = train_epoch_tfmr
    evaluate = evaluate_tfmr
    
elif MODEL == "gcn_gru":
    raise NotImplemented()
    
else:
    raise ValueError("Wrong model choice")

model.apply(initialize_weights)
# training

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=TGT_PAD_IDX)
best_valid_loss = float('inf')
early_stopper = EarlyStopping(MODEL_PATH, patience=PATIENCE)
logger = Logger(LOG_PATH, append_time=False)

for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss = train_epoch(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    early_stopper(valid_loss, model)
    learning_rate_decay(optimizer, LR_DECAY_RATIO, 1e-5)
    if early_stopper.early_stop:
        break
    print_status(logger, epoch, epoch_mins, epoch_secs, train_loss, valid_loss)
    
    
# testing
model.load_state_dict(torch.load(MODEL_PATH))
test_loss = evaluate(model, test_iterator, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
logger.write(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

