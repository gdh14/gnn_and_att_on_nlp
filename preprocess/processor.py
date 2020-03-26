import yaml
import os
import sys
from nose.tools import assert_equal, assert_true



class Processor():
    def __init__(self, config_file):
        self.config = yaml.load(config_file)
        self.label2idx = {}
        self.doc_words_ls = []
        self.label_ls = []

    def load_data(data_dir):
        pass

    def get_label2idx(self):
        if len(self.label2idx) == 0:
            raise ValueError("Label2Idx not generated yet.")
        return self.label2idx

    def get_doc_words_ls(self):
        return self.doc_words_ls

    def get_label_ls(self):
        return self.label_ls

    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def shuffle_(self, random_seed):
        


class TextClassficationProcessor(Processor):
    def __init__(config_file):
        super().__init__(config_file):

    def load_data(data_dir, cls_task):
        data = {}
        if cls_task == '20NG':
            doc_words_ls, label_ls = self._load_data_from_folder(data_dir)
            data['feature'] = doc_words_ls
            data['label'] = label_ls
        return data

    def _load_data_from_folder(data_dir):
        all_clean_doc_ls = []
        all_label_ls = []
        labels = os.listdir(data_dir)
        self.label2idx = self._get_label2idx_from_label_ls(labels)

        for label in labels:
            doc_idx_ls = os.listdir(os.path.join(data_dir, label))
            for doc_idx in doc_idx_ls:
                doc_dir = os.path.join(data_dir, label, doc_idx)
                doc = []
                with open(doc_dir, 'r') as f:
                    for line in f:
                        doc.append(line.strip())
                doc = " ".join(doc)
                clean_doc = self.clean_str(doc)
                all_clean_doc_ls.append(clean_doc)
                all_label_ls.append(self.label2idx[label])

        return all_clean_doc_ls, all_label_ls

    def _get_label2idx_from_label_ls(self, label_ls):
        label2idx = {}
        for idx, label in enumerate(label_ls):
            label2idx[label] = idx
        return label2idx

if __name__ == '__main__':
    config_dir = '../config.yaml'
    train_data_dir = '../data/20NG/20news-bydate-train'
    test_data_dir = '../data/20NG/20news-bydate-test'

    processor = TextClassficationProcessor('../config.yaml')
    train_data = processor.load_data(train_data_dir, '20NG')
    test_data = processor.load_data(train_data_dir, '20NG')




