#!/bin/bash

SAVE_ID=$1
python train.py --model rnn --data_dir dataset/semeval --vocab_dir dataset/vocab --id $SAVE_ID --info "Position-aware attention model"
