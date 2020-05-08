Relation Extraction
==========

This folder contains the source code to run Graph Convolutional Netork as well as Atention Based RNNs for Relation Extraction.

## Requirements

- Python 3 (tested on 3.7.4)
- PyTorch (tested on 0.4.0)
- Stanza (tested on 1.0.1)

## Preparation

Source code performs relation extraction on [SevEval Task 8 Dataset](http://www.kozareva.com/downloads.html). It's already contained in the repo in `dataset/raw`.

To run NLP preprocessing like POS tagging, dependency parsing, etc, run
```
python process_semeval_data.py
```
It will generate preprocessed SemEval dataset in `dataset/semeval`


Besides, Glove Embedding is also required, use the following command to download the Glove word embedding
```
bash download.sh
```

Then prepare vocabulary and initial word vectors with:
```
python prepare_vocab.py dataset/semeval dataset/vocab --glove_dir dataset/glove
```

This will write vocabulary and word vectors as a numpy matrix into the dir `dataset/vocab`.

## Training

To train a graph convolutional neural network (GCN) model, run:
```
bash train_gcn.sh gcn_0
```

Model checkpoints and logs will be saved to `./saved_models/gcn_0`.

To train LSTM + GCN model, run:
```
bash train_lstm_gcn.sh lstm_gcn_0
```

To train LSTM model, run:
```
bash train_lstm.sh lstm_0
```

To train LSTM + Attention model, run:
```
bash train_lstm_att.sh lstm_att_0
```


## Reference
- [Graph Convolution over Pruned Dependency Trees Improves Relation Extraction (authors' PyTorch implementation)](https://github.com/qipeng/gcn-over-pruned-trees)
- [PyTorch implementation of the position-aware attention model for relation extraction](https://github.com/yuhaozhang/tacred-relation)
-  [A simple tool for converting data format from SemEval2010 to TACRED](https://github.com/onehaitao/SemEval2010-to-TACRED)

