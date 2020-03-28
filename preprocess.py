"""
Preprocessing script, need to be further optimized.
"""

import os
import yaml
import numpy as np
import scipy.sparse as sp

from time import time
from preprocess.utils import TextConverter, IOHelper
from preprocess.data import TextDataForTC
from preprocess.build_feature import TextHeteroGraphBuilder

def main():
    start = time()
    # parse config
    config_file = 'config.yaml'
    config = yaml.load(open(config_file), Loader=yaml.FullLoader)
    data_root_dir = config['preprocess']['data_root_dir']
    dataset = config['preprocess']['dataset']

    # generate text data
    text_data = TextDataForTC(config_file, '20ng')

    # generate feature and label data [x, y, tx, ty, allx, ally]
    label2idx = text_data.get_label2idx()
    vocab = text_data.get_vocab()
    word2idx = vocab.get_word2idx()
    text_ls_train = text_data.get_text_ls('train')
    text_ls_val = text_data.get_text_ls('val')
    text_ls_test = text_data.get_text_ls('test')
    label_ls_train = text_data.get_label_ls('train')
    label_ls_val = text_data.get_label_ls('val')
    label_ls_test = text_data.get_label_ls('test')

    feature_train = TextConverter.build_doc_feature(text_ls_train, vocab)
    feature_val = TextConverter.build_doc_feature(text_ls_val, vocab)
    feature_test = TextConverter.build_doc_feature(text_ls_test, vocab)
    vocab_embedding = vocab.get_word_embedding()
    feature_all = np.vstack([feature_train, feature_val, vocab_embedding])
    feature_all = sp.csr_matrix(feature_all)

    label_train = TextConverter.build_onehot_label(label_ls_train, label2idx)
    label_val = TextConverter.build_onehot_label(label_ls_val, label2idx)
    label_test = TextConverter.build_onehot_label(label_ls_test, label2idx)
    label_vocab = np.zeros((len(vocab), len(label2idx)))
    label_all = np.vstack((label_train, label_val, label_vocab))

    IOHelper.save_file(feature_train,
            os.path.join(data_root_dir, "ind.{}.x".format(dataset)), 'pickle')
    IOHelper.save_file(feature_val,
            os.path.join(data_root_dir, "ind.{}.vx".format(dataset)), 'pickle')
    IOHelper.save_file(feature_test,
            os.path.join(data_root_dir, "ind.{}.tx".format(dataset)), 'pickle')    
    IOHelper.save_file(feature_all,\
            os.path.join(data_root_dir, "ind.{}.allx".format(dataset)), 'pickle')
    IOHelper.save_file(label_train,\
            os.path.join(data_root_dir, "ind.{}.y".format(dataset)), 'pickle')
    IOHelper.save_file(label_test,\
            os.path.join(data_root_dir, "ind.{}.ty".format(dataset)), 'pickle')    
    IOHelper.save_file(label_all,\
            os.path.join(data_root_dir, "ind.{}.ally".format(dataset)), 'pickle')    

    print("feature train shape {}".format(feature_train.shape))
    print("feature val shape {}".format(feature_val.shape))
    print("feature test shape {}".format(feature_test.shape))
    print("feature all shape {}".format(feature_all.shape))
    print("label train shape {}".format(label_train.shape))
    print("label test shape {}".format(label_test.shape))
    print("label all shape {}".format(label_all.shape))

    # build the heterograph (edge weights)
    graph_builder = TextHeteroGraphBuilder(config_file, train_size=len(text_ls_train), 
            val_size=len(text_ls_val), test_size=len(text_ls_test), vocab_size=len(vocab))

    text_ls = text_ls_train + text_ls_val + text_ls_test
    hetero_graph = graph_builder.build_graph(text_ls, word2idx)

    IOHelper.save_file(hetero_graph,\
        os.path.join(data_root_dir, "ind.{}.adj".format(dataset)), 'pickle')    
    
    print("Preprocessing finished [{:.0f}s]".format(time() - start))

if __name__ == '__main__':
    main()
