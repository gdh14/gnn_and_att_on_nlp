#!/bin/bash

# Training

###### gcn vs. gru ######
# python main.py hyperparams/gcn_gru1.yaml
# python main.py hyperparams/gcn_gru2.yaml
# python main.py hyperparams/gcn_gru3.yaml
# python main.py hyperparams/gcn_gru4.yaml

# python main.py hyperparams/gcngru_gru1.yaml
python main.py hyperparams/gcngru_gru2.yaml
# python main.py hyperparams/gcngru_gru3.yaml
python main.py hyperparams/gcngru_gru4.yaml

# python main.py hyperparams/gru_seq2seq1.yaml
# python main.py hyperparams/gru_seq2seq2.yaml
# python main.py hyperparams/gru_seq2seq3.yaml
# python main.py hyperparams/gru_seq2seq4.yaml


###### gcu vs. transformer ######
# python main.py hyperparams/transformer1.yaml
# python main.py hyperparams/transformer2.yaml
# python main.py hyperparams/transformer3.yaml
# python main.py hyperparams/transformer4.yaml
# python main.py hyperparams/transformer5.yaml   # best


###### gcn vs. gru attention ######
# python main.py hyperparams/gru_attn1.yaml
# python main.py hyperparams/gru_attn2.yaml
# python main.py hyperparams/gru_attn3.yaml
# python main.py hyperparams/gru_attn4.yaml


# Testing  TO DEBUG
# python test.py hyperparams/gru_attn1.yaml
# python test.py hyperparams/transformer1.yaml
