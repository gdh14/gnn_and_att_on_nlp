Text Classification
===========

## Prepare Your data
1. Download [20 NG Dataset](http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz)
2. Rename the folder as `20ng`
3. Save the folder into `data` folder

## Training and Evaluation Process (Model borrows from torch_gcn.pytorch)

1. Run `python preprocess.py config_20ng.yaml`
2. Run `python train.py 20ng`
>>>>>>> 9289fa69ba0a39f7e179202255db174ce05764b8:text_classification/README.md
