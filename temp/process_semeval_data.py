#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import json
import re
import stanfordnlp
import stanza 
import os

from tqdm import tqdm
from collections import Counter

nlp = stanza.Pipeline(lang='en', tokenize_pretokenized=True)

def relation_process(raw_relation):
    relation = raw_relation.strip()
    subj_type = 'Other'
    obj_type = 'Other'
    pattern = r'(.*?)-(.*?)\((e[12]),(e[12])\)'
    if relation != 'Other':
        value = re.findall(pattern, relation)[0]
        if value[2] == 'e1':
            subj_type == value[0]
            obj_type = value[1]
        else:
            subj_type = value[1]
            obj_type = value[0]
    res = dict(
        relation=relation,
    )
    return res

def clean_sentence(raw_sentence):
    sentence = raw_sentence[1:-1]  # remove quotes
    sentence = re.sub(r"(</*e[12]>)", r" \1 ", sentence)
    sentence = re.sub(r"([,.?!])", r" \1 ", sentence)
    return sentence.strip()

def get_start_end(token_ls):
    subj_start, subj_end, obj_start, obj_end = None, None, None, None
    shift = 0

    for idx, token in enumerate(token_ls):
        idx -= shift
        if token == '<e1>':
            subj_start = idx
        elif token == '</e1>':
            subj_end = idx - 2
            shift = 2
        elif token == '<e2>':
            obj_start = idx
        elif token == '</e2>':
            obj_end = idx - 2
            shift = 2

    assert subj_start <= subj_end
    assert obj_start <= obj_end

    return subj_start, subj_end, obj_start, obj_end


def sentence_process(raw_sentence, raw_relation):
    sentence = clean_sentence(raw_sentence)
    token = sentence.split()

    subj_start, subj_end, obj_start, obj_end = get_start_end(token)

    assert '<e1>' in token
    assert '<e2>' in token
    assert '</e1>' in token
    assert '</e2>' in token
    token.remove('<e1>')
    token.remove('<e2>')
    token.remove('</e1>')
    token.remove('</e2>')

    doc = nlp([token])

    pos = [token.xpos for sent in doc.sentences for token in sent.words]
    ner = [token.ner for sent in doc.sentences for token in sent.tokens]
    deprel = [token.deprel for sent in doc.sentences for token in sent.words]
    head = [str(token.head) for sent in doc.sentences for token in sent.words]
    subj_type = most_common_ner(ner[subj_start : subj_end + 1])
    obj_type = most_common_ner(ner[obj_start : obj_end + 1])

    assert len(token) == len(pos) == len(ner) == len(deprel) == len(head)

    relation = raw_relation.strip()

    res = dict(
        relation=relation,
        token=token,
        subj_start=subj_start,
        subj_end=subj_end,
        obj_start=obj_start,
        obj_end=obj_end,
        subj_type=subj_type,
        obj_type=obj_type,
        stanford_pos=pos,
        stanford_ner=ner,
        stanford_head=head,
        stanford_deprel=deprel
    )

    return res

def most_common_ner(ners):
    ner_counter = Counter(ners)
    most_common_ls = ner_counter.most_common()
    if len(most_common_ls) > 1:
        if 'O' == most_common_ls[0][0]:
            most_common_ner = 'O'
        else:
            most_common_ner = most_common_ls[1][0]
    else:
        most_common_ner = most_common_ls[0][0]
    if most_common_ner != 'O':
        breakpoint()
    return most_common_ner

def convert(src_file, des_file):
    with open(src_file, 'r', encoding='utf-8') as fr:
        file_data = fr.readlines()

    data = []
    for i in tqdm(range(0, len(file_data), 4)):
        meta = {}
        s = file_data[i].strip().split('\t')
        assert len(s) == 2
        meta['id'] = s[0]
        sen_res = sentence_process(s[1], file_data[i+1])
        data.append({**meta, **sen_res})

    breakpoint()
    with open(des_file, 'w', encoding='utf-8') as fw:
        json.dump(data, fw, ensure_ascii=False)


if __name__ == '__main__':
    if os.isdir('dataset/semeval'):
        os.makedirs('datset/semeval')
    train_src = 'dataset/raw/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'
    test_src = 'dataset/raw/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'
    train_des = 'dataset/semeval/train.json'
    dev_des = 'dataset/semeval/dev.json'
    test_des = 'dataset/semeval/test.json'
    convert(train_src, train_des)
    convert(test_src, dev_des)
    convert(test_src, test_des)

