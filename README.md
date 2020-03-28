# gnn_and_att_on_nlp
A Comparison between Graph and Attention Neural Models on Essential NLP Tasks.

## Prepare Your data
1. Download [20 NG Dataset](http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz)
2. Rename the folder as `20ng`
3. Save the folder into `data` folder

## Training and Evaluation Process (Model borrows from torch_gcn.pytorch)

1. Run `python preprocess.py`
2. Run `python train.py 20ng`
