import yaml
import numpy as np
import scipy.sparse as sp
import pickle as pkl
import os

from preprocess.utils import DocumentStatsBuilder as DSBuilder
from preprocess.utils import TextConverter
from preprocess.data import TextDataForTC
from time import time

class TextGraphBuilder():
    def __init__(self, config_file):
        self._parse_config(yaml.load(open(config_file), Loader=yaml.FullLoader))
        self.weight_ls, self.row_idx_ls, self.col_idx_ls = [], [], []
        self.mark = ""
        self.graph = None

    def _parse_config(self, config):
        self.window_size_for_PMI = config['preprocess']['window_size_for_PMI']
        self.word_word_edge_weight = config['preprocess']['word_word_edge_weight']
        self.doc_word_edge_weight = config['preprocess']['doc_word_edge_weight']
        self.min_PMI = config['preprocess']['min_PMI']
        self.graph_root_dir = config['preprocess']['graph_root_dir']
        self.dataset = config['preprocess']['dataset']


class TextHeteroGraphBuilder(TextGraphBuilder):
    def __init__(self, config_file, **kargs):
        super().__init__(config_file)
        self.kargs = kargs
        self.mark = "_hetero"
    
    def build_graph(self, text_ls, word2idx):
        word_word_edge_cnt = self._build_word_word_edge(text_ls, word2idx)
        doc_word_edge_cnt = self._build_doc_word_edge(text_ls, word2idx)
        node_size = self.kargs['train_size'] + self.kargs['val_size'] +\
                    self.kargs['test_size'] + self.kargs['vocab_size']
        self.graph = sp.csr_matrix((self.weight_ls, (self.row_idx_ls, self.col_idx_ls)), 
                                 shape=(node_size, node_size))
        
        print('word-word edges in the graph: {:d}'.format(word_word_edge_cnt))
        print('doc-word edges in the graph: {:d}'.format(doc_word_edge_cnt))
        print('total edges in the graph: {:d}'.format(self.graph.nnz))
        return self.graph

    def _build_word_word_edge(self, text_ls, word2idx, weight_solution='PMI'):
        start = time()
        if self.word_word_edge_weight == 'PMI':
            edge_cnt = self._build_word_word_edge_with_PMI(text_ls, word2idx)
        else:
            raise ValueError("Weight solution {} is not defined".format(weight_solution))
        print('word-word edge built. [{:.0f}s]'.format(time() - start))
        return edge_cnt

    def _build_doc_word_edge(self, text_ls, word2idx):
        start = time()

        train_val_size = self.kargs['train_size'] + self.kargs['val_size']
        vocab_size = len(word2idx)

        tf_idf = DSBuilder.TF_IDF(text_ls, word2idx)
        for (doc_i, word_j) in tf_idf:
            self.weight_ls.append(tf_idf[(doc_i, word_j)])
            if doc_i < train_val_size:
                self.row_idx_ls.append(doc_i)
            else:
                self.row_idx_ls.append(doc_i + vocab_size)
            self.col_idx_ls.append(word_j + train_val_size)

        print('doc-word edge built. [{:.0f}s]'.format(time() - start))
        edge_cnt = len(tf_idf)
        return edge_cnt

    def _build_word_word_edge_with_PMI(self, text_ls, word2idx):
        shift = self.kargs['train_size'] + self.kargs['val_size']
        pmi = DSBuilder.PMI(text_ls, word2idx, self.window_size_for_PMI)
        for (w_i, w_j) in pmi:
            if pmi[(w_i, w_j)] > self.min_PMI:
                self.weight_ls.append(pmi[(w_i, w_j)])
                self.row_idx_ls.append(w_i + shift)
                self.col_idx_ls.append(w_j + shift)
        edge_cnt = len(self.row_idx_ls)
        return edge_cnt

if __name__ == '__main__':
    config_file = '../config.yaml'
    text_data = TextDataForTC(config_file, '20ng')
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
    feature_test = TextConverter.build_doc_feature(text_ls_test, vocab)
    label_train_onehot = TextConverter.build_onehot_label(label_ls_train, label2idx)
    label_test_onehot = TextConverter.build_onehot_label(label_ls_test, label2idx)

    graph_builder = TextHeteroGraphBuilder(config_file, train_size=len(text_ls_train), 
            val_size=len(text_ls_val), test_size=len(text_ls_test), vocab_size=len(vocab))

    text_ls = text_ls_train + text_ls_val + text_ls_test
    graph_builder.build_graph(text_ls, word2idx)
    graph_builder.save_graph_data()
