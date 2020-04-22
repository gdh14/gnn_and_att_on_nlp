#!/bin/bash

# run all
# python3 train.py config_20ng_lstm.yaml
# python3 test.py config_20ng_lstm.yaml

# python3 train.py config_20ng_tfmr.yaml
# python3 test.py config_20ng_tfmr.yaml

# python3 train.py config_mr_lstm.yaml
# python3 test.py config_mr_lstm.yaml

# python3 train.py config_mr_tfmr.yaml
# python3 test.py config_mr_tfmr.yaml

# hyper-param tuning
python3 train.py model_tuning/config_20ng_tfmr1.yaml
python3 test.py model_tuning/config_20ng_tfmr1.yaml

python3 train.py model_tuning/config_20ng_tfmr2.yaml
python3 test.py model_tuning/config_20ng_tfmr2.yaml

python3 train.py model_tuning/config_20ng_tfmr3.yaml
python3 test.py model_tuning/config_20ng_tfmr3.yaml

python3 train.py model_tuning/config_20ng_tfmr4.yaml
python3 test.py model_tuning/config_20ng_tfmr4.yaml