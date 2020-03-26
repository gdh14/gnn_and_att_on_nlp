import math as m
from nose.tools import assert_equal, assert_true


"""
A class to get document statistics, like tf-idf for NLP tasks.
"""
class DocumentStatsBuilder():
    """
    Build the inverted index for a list of document.

    Args:
        doc_words_ls: A list of document, each document is a string of words seperated
                      by space.
        word2idx: A dictm mapping from word to index.
    Return: 
        a dict represents the inverted index, both the word and the docs are
        in index format (numericalized).
        Example: {
            word_1_idx: [doc_1_idx, doc_2_idx, doc_4_idx]
            word_2_idx: [doc_2, doc_3]
        }
    """
    @staticmethod
    def build_inverted_index(doc_words_ls, word2idx):
        inverted_index = {}        
        for doc_idx, doc_words in enumerate(doc_words_ls):
            appeared = set()
            for word in doc_words.split():
                if word in appeared:
                    continue
                word_idx = word2idx[word]
                if word_idx not in inverted_index:
                    inverted_index[word_idx] = [doc_idx]
                else:
                    inverted_index[word_idx].append(doc_idx)
                appeared.add(word)
        
        return inverted_index

    @staticmethod
    def doc_freq(doc_words_ls, word2idx, inverted_index=None):
        doc_freq = {}

        if inverted_index is None:
            inverted_index = DocumentStatsBuilder.build_inverted_index(doc_words_ls, word2idx)

        for k, v in inverted_index.items():
            doc_freq[k] = len(v)

        return doc_freq

    """
    (doc_id, term_id): tf_val
    """
    @staticmethod
    def term_freq(doc_words_ls, word2idx, normalize=False):
        def get_normalized_tf_value(word):
            word_idx = word2idx[word]
            if normalize:
                return (doc_idx, word_idx, 1 / word_cnt)
            else:
                return (doc_idx, word_idx, 1)

        term_freq = {}
        for doc_idx, doc_words in enumerate(doc_words_ls):
            words_ls = doc_words.split()
            word_cnt = len(words_ls)
            tf_temp_ls = list(map(lambda word: get_normalized_tf_value(word), words_ls))
            for tf_temp in tf_temp_ls:
                doc_word_pair = tf_temp[:2]
                tf_val = tf_temp[2]
                if doc_word_pair in term_freq:
                    term_freq[doc_word_pair] += tf_val
                else:
                    term_freq[doc_word_pair] = tf_val
        return term_freq

    @staticmethod
    def _get_doc_windows(doc_words_ls, window_size):
        doc_windows = []

        for doc_words in doc_words_ls:
            words = doc_words.split()
            length = len(words)
            if length <= window_size:
                doc_windows.append(words)
            else:
                for i in range(length - window_size + 1):
                    window = words[i: i + window_size]
                    doc_windows.append(window)
        
        return doc_windows

    @staticmethod
    def _get_word_window_freq(doc_windows, word2idx):
        word_window_freq = {}

        for window in doc_windows:
            appeared = set()
            for word in window:
                word_idx = word2idx[word]
                if word_idx in appeared:
                    continue
                if word_idx in word_window_freq:
                    word_window_freq[word_idx] += 1
                else:
                    word_window_freq[word_idx] = 1
                appeared.add(word_idx)

        return word_window_freq

    @staticmethod
    def _get_word_pair_cnt(doc_windows, word2idx):
        word_pair_count = {}

        for window in doc_windows:
            for i in range(1, len(window)):
                for j in range(0, i):
                    word_i_id = word2idx[window[i]]
                    word_j_id = word2idx[window[j]]

                    # ignore when two words are the same
                    if word_i_id == word_j_id:
                        continue

                    word_pair = (word_i_id, word_j_id)
                    if word_pair in word_pair_count:
                        word_pair_count[word_pair] += 1
                    else:
                        word_pair_count[word_pair] = 1
                    
                    # reverse order
                    word_pair = (word_j_id, word_i_id)
                    if word_pair in word_pair_count:
                        word_pair_count[word_pair] += 1
                    else:
                        word_pair_count[word_pair] = 1

        return word_pair_count

    @staticmethod
    def PMI(doc_words_ls, word2idx, window_size):
        if window_size < 2:
            raise ValueError("window size must be greater than 1.")
 
        PMI = {}
        doc_windows = DocumentStatsBuilder._get_doc_windows(doc_words_ls, window_size)
        word_window_freq = DocumentStatsBuilder._get_word_window_freq(doc_windows, word2idx)
        word_pair_cnt = DocumentStatsBuilder._get_word_pair_cnt(doc_windows, word2idx) 

        for (i, j), pair_cnt_ij in word_pair_cnt.items():
            PMI[(i, j)] = m.log(pair_cnt_ij * len(doc_windows) / 
                (word_window_freq[i] * word_window_freq[j]))

        return PMI


"""
Simple unit test.
"""
if __name__ == '__main__':
    doc_words_ls = ['I I like Zoe I', 'Zoe like me me']
    word2idx = {'I': 0, 'like': 1, 'Zoe': 2, 'me':3}

    # test build_inverted_index
    expected = {
        0: [0],
        1: [0, 1],
        2: [0, 1],
        3: [1]
    }
    inverted_index = DocumentStatsBuilder.build_inverted_index(doc_words_ls, word2idx)
    assert_equal(expected, inverted_index, "Incorrect result for DocumentStatsBuilder.build_inverted_index")
    
    # test doc_freq
    expected = {0: 1, 1: 2, 2: 2, 3: 1}
    assert_equal(expected, DocumentStatsBuilder.doc_freq(doc_words_ls, word2idx, inverted_index), 
            "Incorrect result for DocumentStatsBuilder.doc_freq")
    assert_equal(expected, DocumentStatsBuilder.doc_freq(doc_words_ls, word2idx), 
            "Incorrect result for DocumentStatsBuilder.doc_freq")
    
    # test term_freq
    expected = {
        (0, 0): 3,
        (0, 1): 1,
        (0, 2): 1,
        (1, 1): 1,
        (1, 2): 1,
        (1, 3): 2,
    }
    assert_equal(expected, DocumentStatsBuilder.term_freq(doc_words_ls, word2idx, False),
            "Incorrect result for DocumentStatsBuilder.term_freq")

    # test PMI
    expected = {
        (1, 0): -0.54, 
        (0, 1): -0.54, 
        (2, 1): 0.15, 
        (1, 2): 0.15, 
        (0, 2): -0.25, 
        (2, 0): -0.25, 
        (3, 1): -0.13, 
        (1, 3): -0.13
    }
    pmi = DocumentStatsBuilder.PMI(doc_words_ls, word2idx, window_size=2)
    for k, v in pmi.items():
        pmi[k] = round(v, 2)
    assert_equal(expected, pmi, "Incorrect result for DocumentStatsBuilder.PMI")
