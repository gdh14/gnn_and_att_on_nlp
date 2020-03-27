import yaml
import os
import sys
import random
import nltk
import re

from utils import DocumentStatsBuilder as DSBuilder
from nltk.corpus import stopwords
from nose.tools import assert_equal, assert_true

nltk.download('stopwords')

INVALID_FILE_NAME = '.DS_Store'

# class TextTensorData(Dataset):
#     def __init__():

"""
The abstract class for text data.
"""
class TextData():
    def __init__(self, config_file):
        self.parse_config(yaml.load(open(config_file), Loader=yaml.FullLoader))
        self.word_freq, self.label2idx = {}, {}
        self.raw_text_ls_train, self.label_ls_train, self.preprocessed_text_ls_train = [], [], []
        self.raw_text_ls_test, self.label_ls_test, self.preprocessed_text_ls_test = [], [], []        
        self.stopwords = set() if self.remove_stop_words else set(stopwords.words('english'))

    def parse_config(self, config):
        self.remove_stop_words = config['preprocess']['remove_stop_words']
        self.train_random_seed = config['preprocess']['train_random_seed']
        self.test_random_seed = config['preprocess']['test_random_seed']
        self.val_split_ratio = config['preprocess']['val_split_ratio']
        self.train_data_dir = config['preprocess']['train_data_dir']
        self.test_data_dir = config['preprocess']['test_data_dir']
        self.min_word_freq = config['preprocess']['min_word_freq']

    def _get_tensor_data(self, data_type):
        if data_type == 'train':
            text_ls, label_ls = self.preprocessed_text_ls_train, self.label_ls_train
        elif data_type == 'val':
            text_ls, label_ls = self.preprocessed_text_ls_val, self.label_ls_val
        else:
            text_ls, label_ls = self.preprocessed_text_ls_test, self.label_ls_test
        return text_ls, label_ls

    def get_tensor_data_train(self):
        return self._get_tensor_data('train')

    def get_tensor_data_val(self):
        return self._get_tensor_data('val')

    def get_tensor_data_test(self):
        return self._get_tensor_data('test')


    def load_data(self, data_dir, **kargs):
        pass

    def get_label2idx(self):
        if len(self.label2idx) == 0:
            raise ValueError("Label2Idx not generated yet.")
        return self.label2idx

    def get_text_ls(self):
        return self.data['text']

    def get_label_ls(self):
        return self.data['label']

    def clean_str(self, string):
        return DSBuilder.clean_str(string)

    def preprocess_string(self, string):
        preprocessed_string = self.clean_str(string)
        words = preprocessed_string.split()
        preprocessed_string_ls = []
        
        for word in words:
            if self.word_freq[word] >= self.min_word_freq and\
                    word not in self.stopwords:
                preprocessed_string_ls.append(word)

        preprocessed_string = " ".join(preprocessed_string_ls)
        return preprocessed_string

    def _split_and_shuffle(self):
        # shuffle the index
        train_idx = list(range(len(self.label_ls_train)))
        test_idx = list(range(len(self.label_ls_test)))
        random.seed(self.train_random_seed)
        random.shuffle(train_idx)
        random.seed(self.test_random_seed)
        random.shuffle(test_idx)
        
        # split the train set
        train_size = int(len(train_idx) * (1 - self.val_split_ratio))
        val_idx = train_idx[train_size:]
        train_idx = train_idx[:train_size]

        # shuffle the data
        self.raw_text_ls_val = [self.raw_text_ls_train[i] for i in val_idx]
        self.label_ls_val = [self.label_ls_train[i] for i in val_idx]
        self.preprocessed_text_ls_val = [self.preprocessed_text_ls_train[i] for i in val_idx]
        self.raw_text_ls_train = [self.raw_text_ls_train[i] for i in train_idx]
        self.label_ls_train = [self.label_ls_train[i] for i in train_idx]
        self.preprocessed_text_ls_train = [self.preprocessed_text_ls_train[i] for i in train_idx]
        self.raw_text_ls_test = [self.raw_text_ls_test[i] for i in test_idx]
        self.label_ls_test = [self.label_ls_test[i] for i in test_idx]
        self.preprocessed_text_ls_test = [self.preprocessed_text_ls_test[i] for i in test_idx]

    def get_word_freq(self, text_ls):
        return DSBuilder.word_freq(text_ls)

    def preprocess_text_ls(self, text_ls):
        preprocessed_text_ls = []

        for text in text_ls:
            preprocessed_text_ls.append(self.preprocess_string(text))

        return preprocessed_text_ls

"""
Text data for text classification data
"""
class TextDataForTC(TextData):
    def __init__(self, config_file, cls_task):
        super().__init__(config_file)
        self.raw_text_ls_train, self.label_ls_train = self.load_data(
                self.train_data_dir, cls_task=cls_task)
        self.raw_text_ls_test, self.label_ls_test = self.load_data(
                self.test_data_dir, cls_task=cls_task)

        # word frequency build on raw_text_ls_train and raw_text_ls_test??
        # probably not correct
        self.word_freq = self.get_word_freq(self.raw_text_ls_train + self.raw_text_ls_test)

        self.preprocessed_text_ls_train = self.preprocess_text_ls(self.raw_text_ls_train)
        self.preprocessed_text_ls_test = self.preprocess_text_ls(self.raw_text_ls_test)

        self._split_and_shuffle()

    def load_data(self, data_dir, **kargs):
        if kargs['cls_task'] == '20ng':
            text_ls, label_ls = self._load_data_from_folder(data_dir)
        return text_ls, label_ls

    def _load_data_from_folder(self, data_dir):
        all_text_ls = []
        all_label_ls = []
        labels = [x for x in os.listdir(data_dir) if x != INVALID_FILE_NAME]
        self.label2idx = self._get_label2idx_from_label_ls(labels)

        for label in labels:
            doc_idx_ls = os.listdir(os.path.join(data_dir, label))
            for doc_idx in doc_idx_ls:
                doc_dir = os.path.join(data_dir, label, doc_idx)
                line_ls = []
                with open(doc_dir, 'r', encoding='latin1') as f:
                    for line in f:
                        line_ls.append(line.strip())
                doc_text = " ".join(line_ls)
                all_text_ls.append(doc_text)
                all_label_ls.append(self.label2idx[label])

        return all_text_ls, all_label_ls

    def _get_label2idx_from_label_ls(self, label_ls):
        label2idx = {}
        for idx, label in enumerate(label_ls):
            label2idx[label] = idx
        return label2idx

if __name__ == '__main__':
    config_dir = '../config.yaml'
    train_data_dir = '../data/20NG/20news-bydate-train'
    test_data_dir = '../data/20NG/20news-bydate-test'
    text_data = TextDataForTC('../config.yaml', '20ng')

