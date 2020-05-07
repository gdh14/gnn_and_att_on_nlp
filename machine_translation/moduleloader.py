import spacy, random, math, time, yaml, sys, os
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
get_sentence_lengths, counter2array, ensure_path_exist, \
print_status, learning_rate_decay, count_parameters
from src.early_stopping import EarlyStopping
from src.logging import Logger

# model builder for jupyter notebook

class ModuleLoader(object):
    
    def __init__(self, yaml_path):
        config_file = yaml_path
        config = yaml.load(open(config_file), Loader=yaml.FullLoader)
        args = config["training"]
        SEED = args["seed"]
        DATASET = args["dataset"]  # Multi30k or ISWLT
        MODEL = args["model"]  # gru**2, gru_attn**2, transformer, gcn_gru, gcngru_gru, gcngruattn_gru, gcnattn_gru
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
        LOG_PATH = '{}test-log.log'.format(DIR)
        set_seed(SEED)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.config = args
        self.device = device

        if 'transformer' in MODEL:
            ENC_HEADS = args["encoder_heads"]
            DEC_HEADS = args["decoder_heads"]
            ENC_PF_DIM = args["encoder_pf_dim"]
            DEC_PF_DIM = args["decoder_pf_dim"]
            MAX_LEN = args["max_len"]
            
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
        self.train_data, self.valid_data, self.test_data = train_data, valid_data, test_data
        
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
            (train_data, valid_data, test_data), 
            batch_size = BATCH_SIZE, 
            sort_key = lambda x: len(x.src),
            sort_within_batch=False,
            device = device)
        self.train_iterator, self.valid_iterator, self.test_iterator = train_iterator, valid_iterator, test_iterator
        
        SRC.build_vocab(train_data, min_freq = 2)
        TGT.build_vocab(train_data, min_freq = 2)
        self.SRC, self.TGT, self.GRH = SRC, TGT, GRH

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
        self.SRC_PAD_IDX = SRC_PAD_IDX
        self.TGT_PAD_IDX = TGT_PAD_IDX

        if MODEL == "gru**2":  # gru**2, gru_attn**2, transformer, gcn_gru
            from models.gru_seq2seq import GRUEncoder, GRUDecoder, Seq2Seq
            enc = GRUEncoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, NLAYERS, ENC_DROPOUT)
            dec = GRUDecoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, NLAYERS, DEC_DROPOUT)
            model = Seq2Seq(enc, dec, device).to(device)

            from src.train import train_epoch_gru, evaluate_gru, epoch_time
            train_epoch = train_epoch_gru
            evaluate = evaluate_gru
            
            self.enc, self.dec, self.model, self.train_epoch, self.evaluate = enc, dec, model, train_epoch, evaluate
            
        elif MODEL == "gru_attn**2":
            from models.gru_attn import GRUEncoder, GRUDecoder, Seq2Seq, Attention
            attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
            enc = GRUEncoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, NLAYERS, ENC_DROPOUT)
            dec = GRUDecoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, NLAYERS, DEC_DROPOUT, attn)
            model = Seq2Seq(enc, dec, device).to(device)

            from src.train import train_epoch_gru_attn, evaluate_gru_attn, epoch_time
            train_epoch = train_epoch_gru_attn
            evaluate = evaluate_gru_attn
            
            self.enc, self.dec, self.model, self.train_epoch, self.evaluate, self.attn = enc, dec, model, train_epoch, evaluate, attn

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

            self.enc, self.dec, self.model, self.train_epoch, self.evaluate = enc, dec, model, train_epoch, evaluate
            
        elif MODEL == "gcn_gru":
            from models.gru_seq2seq import GCNEncoder, GRUDecoder, GCN2Seq
            enc = GCNEncoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, NLAYERS, ENC_DROPOUT)
            dec = GRUDecoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, NLAYERS, DEC_DROPOUT)
            model = GCN2Seq(enc, dec, device).to(device)

            from src.train import train_epoch_gcn_gru, evaluate_gcn_gru, epoch_time
            train_epoch = train_epoch_gcn_gru
            evaluate = evaluate_gcn_gru

            self.enc, self.dec, self.model, self.train_epoch, self.evaluate = enc, dec, model, train_epoch, evaluate
            
        elif MODEL == "gcngru_gru":
            from models.gru_seq2seq import GCNGRUEncoder, GRUDecoder, GCN2Seq
            enc = GCNGRUEncoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, NLAYERS, ENC_DROPOUT, device)
            dec = GRUDecoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, NLAYERS, DEC_DROPOUT)
            model = GCN2Seq(enc, dec, device).to(device)

            from src.train import train_epoch_gcn_gru, evaluate_gcn_gru, epoch_time
            train_epoch = train_epoch_gcn_gru
            evaluate = evaluate_gcn_gru

            self.enc, self.dec, self.model, self.train_epoch, self.evaluate = enc, dec, model, train_epoch, evaluate
            
        elif MODEL == "gcnattn_gru":
            from models.gru_attn import GCNEncoder, GRUDecoder, GCN2Seq, Attention
            attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
            enc = GCNEncoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, NLAYERS, ENC_DROPOUT)
            dec = GRUDecoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, NLAYERS, DEC_DROPOUT, attn)
            model = GCN2Seq(enc, dec, device).to(device)

            from src.train import train_epoch_gcnattn_gru, evaluate_gcnattn_gru, epoch_time
            train_epoch = train_epoch_gcnattn_gru
            evaluate = evaluate_gcnattn_gru
            
            self.enc, self.dec, self.model, self.train_epoch, self.evaluate, self.attn = enc, dec, model, train_epoch, evaluate, attn

        elif MODEL == "gcngruattn_gru":
            from models.gru_attn import GCNGRUEncoder, GRUDecoder, GCN2Seq, Attention
            attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
            enc = GCNGRUEncoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, NLAYERS, ENC_DROPOUT, device)
            dec = GRUDecoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, NLAYERS, DEC_DROPOUT, attn)
            model = GCN2Seq(enc, dec, device).to(device)

            from src.train import train_epoch_gcnattn_gru, evaluate_gcnattn_gru, epoch_time
            train_epoch = train_epoch_gcnattn_gru
            evaluate = evaluate_gcnattn_gru
            
            self.enc, self.dec, self.model, self.train_epoch, self.evaluate, self.attn = enc, dec, model, train_epoch, evaluate, attn

        else:
            raise ValueError("Wrong model choice")

        if 'gcn' in MODEL:
            from src.utils import init_weights_uniform as init_weights
        else: 
            from src.utils import init_weights_xavier as init_weights

        model.apply(init_weights)
        n_params = count_parameters(model)
        print("Model initialized...{} params".format(n_params))
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=TGT_PAD_IDX)
        
        print(os.path.join(MODEL_PATH, "checkpoint.pt"))
#         try:
#             state_dict = torch.load(os.path.join(MODEL_PATH, "checkpoint.pt"), map_location=device)['model_state_dict']
#         except:
#             state_dict = torch.load(os.path.join(MODEL_PATH, "checkpoint.pt"), map_location=device)
        state_dict = torch.load(os.path.join(MODEL_PATH, "checkpoint.pt"), map_location=device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
        self.model = model