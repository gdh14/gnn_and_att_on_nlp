#!/bin/bash

SAVE_ID=$1
python train.py --model rnn --data_dir dataset/semeval --vocab_dir dataset/vocab --no-attn --id $SAVE_ID \
--info "LSTM model"
