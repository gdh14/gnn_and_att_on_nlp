"""
Define constants.
"""
EMB_INIT_RANGE = 1.0
MAX_LEN = 100

# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

# hard-coded mappings from fields to ids
NER_TO_ID = dict(zip([PAD_TOKEN, UNK_TOKEN, 'O', 'S-GPE', 'E-PERSON', 'S-FAC', 'B-QUANTITY', 'B-TIME', 'E-LAW', 'I-PERSON', 'I-LOC', 'E-CARDINAL', 'E-QUANTITY', 'S-DATE', 'B-MONEY', 'B-EVENT', 'S-LANGUAGE', 'B-ORG', 'B-CARDINAL', 'B-FAC', 'I-CARDINAL', 'B-WORK_OF_ART', 'S-WORK_OF_ART', 'E-PRODUCT', 'E-EVENT', 'B-PERSON', 'S-EVENT', 'B-DATE', 'I-MONEY', 'I-ORG', 'S-PERSON', 'E-MONEY', 'B-LAW', 'E-ORG', 'I-NORP', 'B-PERCENT', 'I-TIME', 'E-PERCENT', 'S-TIME', 'I-DATE', 'S-LOC', 'I-LAW', 'E-DATE', 'S-NORP', 'E-FAC', 'B-NORP', 'E-LOC', 'S-ORDINAL', 'S-LAW', 'B-GPE', 'I-EVENT', 'I-GPE', 'I-FAC', 'S-QUANTITY', 'E-GPE', 'B-PRODUCT', 'E-NORP', 'E-WORK_OF_ART', 'S-CARDINAL', 'E-TIME', 'S-MONEY', 'I-PERCENT', 'I-QUANTITY', 'S-ORG', 'B-LOC', 'S-PRODUCT', 'I-WORK_OF_ART', 'I-PRODUCT'], range(68)))
OBJ_NER_TO_ID = NER_TO_ID
SUBJ_NER_TO_ID = NER_TO_ID

POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'NNP': 2, 'NN': 3, 'IN': 4, 'DT': 5, ',': 6, 'JJ': 7, 'NNS': 8, 'VBD': 9, 'CD': 10, 'CC': 11, '.': 12, 'RB': 13, 'VBN': 14, 'PRP': 15, 'TO': 16, 'VB': 17, 'VBG': 18, 'VBZ': 19, 'PRP$': 20, ':': 21, 'POS': 22, '\'\'': 23, '``': 24, '-RRB-': 25, '-LRB-': 26, 'VBP': 27, 'MD': 28, 'NNPS': 29, 'WP': 30, 'WDT': 31, 'WRB': 32, 'RP': 33, 'JJR': 34, 'JJS': 35, '$': 36, 'FW': 37, 'RBR': 38, 'SYM': 39, 'EX': 40, 'RBS': 41, 'WP$': 42, 'PDT': 43, 'LS': 44, 'UH': 45, '#': 46}

DEPREL_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'punct': 2, 'compound': 3, 'case': 4, 'nmod': 5, 'det': 6, 'nsubj': 7, 'amod': 8, 'conj': 9, 'dobj': 10, 'ROOT': 11, 'cc': 12, 'nmod:poss': 13, 'mark': 14, 'advmod': 15, 'appos': 16, 'nummod': 17, 'dep': 18, 'ccomp': 19, 'aux': 20, 'advcl': 21, 'acl:relcl': 22, 'xcomp': 23, 'cop': 24, 'acl': 25, 'auxpass': 26, 'nsubjpass': 27, 'nmod:tmod': 28, 'neg': 29, 'compound:prt': 30, 'mwe': 31, 'parataxis': 32, 'root': 33, 'nmod:npmod': 34, 'expl': 35, 'csubj': 36, 'cc:preconj': 37, 'iobj': 38, 'det:predet': 39, 'discourse': 40, 'csubjpass': 41}

NEGATIVE_LABEL = 'Other'

LABEL_TO_ID = {
    'Other': 0, 
    'Cause-Effect(e1,e2)': 1, 
    'Cause-Effect(e2,e1)': 2, 
    'Instrument-Agency(e1,e2)': 3, 
    'Instrument-Agency(e2,e1)': 4, 
    'Product-Producer(e1,e2)': 5, 
    'Product-Producer(e2,e1)': 6, 
    'Content-Container(e1,e2)': 7, 
    'Content-Container(e2,e1)': 8, 
    'Entity-Origin(e1,e2)': 9, 
    'Entity-Origin(e2,e1)': 10, 
    'Entity-Destination(e1,e2)': 11, 
    'Entity-Destination(e2,e1)': 12, 
    'Component-Whole(e1,e2)': 13, 
    'Component-Whole(e2,e1)': 14, 
    'Member-Collection(e1,e2)': 15, 
    'Member-Collection(e2,e1)': 16, 
    'Message-Topic(e1,e2)': 17, 
    'Message-Topic(e2,e1)': 18, 
}

INFINITY_NUMBER = 1e12
