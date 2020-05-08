#!/bin/bash

SAVE_ID=$1
python train.py --model gcn --data_dir dataset/semeval --id $SAVE_ID --seed 0 --prune_k 1 --lr 1 \
--no-rnn --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003
