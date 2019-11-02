import json
from collections import OrderedDict
from collections import Counter
import logging
import numpy as np
import collections
import operator
import pickle
from tokenizer import Tokenizer
import re
import random
import glob


def build_dict(documents, questions):
    # create dictionary for entire vocabulary 
    word_count = Counter()
    for sent in questions:
        for w in sent.split():
            word_count[w] += 1
    for sample in documents:
        for utter in sample:
            for w in utter.split():
                word_count[w] += 1

    ls = word_count.most_common()
    logging.info('#Words: %d -> %d' % (len(word_count), len(ls)))
    for key in ls[:5]:
        logging.info(key)
    logging.info('...')
    for key in ls[-5:]:
        logging.info(key)

    return {w[0]: index + 2 for (index, w) in enumerate(ls)}


def vectorize(examples, word_dict, entity_dict, max_d, max_q, max_s, verbose=True):
    in_x1 = []
    in_x2 = []
    in_l = np.zeros((len(examples[0]), len(entity_dict))).astype(np.float32)
    in_y = []
    # masking for actual words in queries
    in_qmask = np.zeros((len(examples[0]), max_q)).astype(np.float32)
    # masking for actual utterances in dialogs
    in_dmask = np.zeros((len(examples[0]), max_s)).astype(np.float32)
    for idx, (d, q, a) in enumerate(zip(examples[0], examples[1], examples[2])):
        q_words = q.split()
        q_vec = np.zeros((max_q))
        for i in range(min(len(q_words), max_q)):
            if q_words[i] in word_dict:
                q_vec[i] = word_dict[q_words[i]]
        scene = []
        for utter in d:
            u_words = utter.split()
            u_vec = np.zeros((max_d))
            for j in range(min(len(u_words), max_d)):
                if u_words[j] in word_dict:
                    u_vec[j] = word_dict[u_words[j]]
                if u_words[j] in entity_dict:
                    in_l[idx, entity_dict[u_words[j]]] = 1.0

            scene.append(u_vec)
        in_dmask[idx, :len(scene)] = 1.0
        while (len(scene) < max_s):
            scene.append(np.zeros((max_d)))
        in_x1.append(scene)
        in_x2.append(q_vec)
        in_y.append(entity_dict[a] if a in entity_dict else 0)
        in_qmask[idx, :min(len(q_words), max_q)] = 1.0

    return in_x1, in_x2, in_l, in_y, in_qmask, in_dmask


def bert_vectorize(examples, token_dict, entity_dict):
    nb_class = len(entity_dict)
    token_inputs = []
    seg_inputs = []

    in_y = []
    e1 = []
    e2 = []
    e3 = []
    for idx, (d, q, a) in enumerate(zip(examples[0], examples[1], examples[2])):
        tokens = ['[CLS]']
        for utter in d:
            u_words = utter.split()
            tokens += u_words
        tokens.append('[SEP]')
        q_words = q.split()
        tokens += q_words
        tokens.append('[SEP]')
        if len(tokens) > 512:
            continue
        e1.append(d)
        e2.append(q)
        e3.append(a)
    in_l = np.zeros((len(e1), len(entity_dict))).astype(np.float32)
    en_ms = []
    en_ds = []
    for idx, (d, q, a) in enumerate(zip(e1, e2, e3)):
        tokens = ['[CLS]']
        en_d = []
        d_words = []
        en_d.append(np.zeros(nb_class))
        d_words.append(np.zeros(768))

        q_words = q.split()
        tokens += q_words
        for j in range(len(q_words)):
            d_words.append(np.zeros(768))
            en_d.append(np.zeros(nb_class))
        tokens.append('[SEP]')
        en_d.append(np.zeros(nb_class))
        d_words.append(np.zeros(768))
        segment_len = len(tokens)

        for utter in d:
            u_words = utter.split()
            tokens += u_words
            for j in range(len(u_words)):
                if u_words[j] in entity_dict:
                    d_words.append(np.ones(768))
                    enen = np.zeros(nb_class)
                    index = int(''.join(filter(str.isdigit, u_words[j])))
                    enen[index + 1] = 1.0
                    en_d.append(enen)
                    in_l[idx, index + 1] = 1.0
                else:
                    d_words.append(np.zeros(768))
                    en_d.append(np.zeros(nb_class))
        tokens.append('[SEP]')
        en_d.append(np.zeros(nb_class))
        d_words.append(np.zeros(768))

        for j in range(512 - len(tokens)):
            d_words.append(np.zeros(768))
            en_d.append(np.zeros(nb_class))

        token_input = []
        for token in tokens:
            if token in token_dict:
                token_input.append(token_dict[token])
            elif token in entity_dict:
                index = int(''.join(filter(str.isdigit, token)))
                token_input.append(index + 1)
            elif token == "placeholder":
                token_input.append(len(entity_dict) + 3)
            else:
                token_input.append(100)
        for j in range(512 - len(tokens)):
            token_input.append(0)
        token_input = np.asarray(token_input)
        seg_input = np.asarray([0] * segment_len + [1] * (len(tokens) - segment_len) + [0] * (512 - len(tokens)))
        token_inputs.append(token_input)
        seg_inputs.append(seg_input)
        in_y.append(int(''.join(filter(str.isdigit, a))) + 1 if a in entity_dict else 0)
        en_m = np.array(d_words)
        en_d = np.array(en_d)
        en_ms.append(en_m)
        en_ds.append(en_d)
    return token_inputs, seg_inputs, in_l, in_y, en_ms, en_ds


def build_match(embeddings, examples, word_dict, max_d, max_q, max_s):
    all_matches = []
    for idx, (d, q, a) in enumerate(zip(examples[0], examples[1], examples[2])):
        q_words = q.split()
        q_vec = [word_dict[w] if w in word_dict else 0 for w in q_words]
        question = np.array([embeddings[w] for w in q_vec])
        sample = []
        for utter in d:
            u_words = utter.split()
            u_vec = [word_dict[w] if w in word_dict else 0 for w in u_words]
            document = np.array([embeddings[w] for w in u_vec])
            score = np.zeros((max_d, max_q))
            for i in range(min(document.shape[0], max_d)):
                for j in range(min(question.shape[0], max_q)):
                    score[i, j] = 1 / (1 + np.linalg.norm(document[i, :] - question[j, :]))
            sample.append(score)
        all_matches.append(sample)
        if idx % 100 == 0:
            print(idx)
    return all_matches


def prune_data(in_file):
    with open(in_file) as data_file:
        samples = json.load(data_file)
    # create prune dictionaries based on document frequency of words
    doc_frequency = Counter()
    for sample in samples:
        question = sample['query'].lower()
        document = []
        for utter in sample['utterances']:
            document.append(' '.join([utter['speakers'], utter['tokens'].lower()]).strip())
        q_words = question.split()
        sample_dict = Counter()
        for w in q_words:
            if w not in sample_dict:
                sample_dict[w] = 1
        for u in document:
            u_words = u.split()
            for w in u_words:
                if w not in sample_dict:
                    sample_dict[w] = 1
        doc_frequency.update(sample_dict)

    sorted_doc_frequency = OrderedDict(sorted(doc_frequency.items(), key=operator.itemgetter(1), reverse=True))
    temp_1 = list(sorted_doc_frequency.keys())[:int(len(doc_frequency) * 0.05)]
    redundent_1 = {k: sorted_doc_frequency[k] for k in temp_1}
    temp_2 = list(sorted_doc_frequency.keys())[:int(len(doc_frequency) * 0.30)]
    redundent_2 = {k: sorted_doc_frequency[k] for k in temp_2}
    return redundent_1, redundent_2


def load_jsondata(in_file, redundent_1, redundent_2, stopwords_file):
    stopwords = []
    with open(stopwords_file) as st:
        for line in st:
            stopwords.append(line.strip())
    with open(in_file) as data_file:
        samples = json.load(data_file)
    documents = []
    questions = []
    answers = []

    max_d_len = 0
    max_q_len = 0
    max_s_len = 0

    for sample in samples:
        question = sample['query'].lower()
        answer = sample['answer']
        document = []
        for utter in sample['utterances']:
            document.append(' '.join([utter['speakers'], utter['tokens'].lower()]).strip())

        if len(document) > max_s_len:
            max_s_len = len(document)

        q_words = question.split()
        if len(q_words) > max_q_len:
            max_q_len = len(q_words)
        d_words = []
        new_document = []
        for u in document:
            u_words = u.split()
            if len(u_words) > 80:
                u_words = [w for w in u_words if w not in stopwords]

            if len(u_words) > 80:
                l = []
                for w in u_words:
                    if (w.startswith('@ent')):
                        l.append(w)
                    elif w not in redundent_1:
                        l.append(w)
                u_words = l
            if len(u_words) > 80:
                l = []
                for w in u_words:
                    if (w.startswith('@ent')):
                        l.append(w)
                    elif w not in redundent_2:
                        l.append(w)
                u_words = l
            d_words += u_words
            new_document.append(u_words)
            if len(u_words) > max_d_len:
                max_d_len = len(u_words)

        entity_dict = {}
        entity_id = 0
        for word in d_words + q_words:
            if (word.startswith('@ent')) and (word not in entity_dict):
                entity_dict[word] = '@ent' + str(entity_id)
                entity_id += 1
        q_words = [entity_dict[w] if w in entity_dict else w for w in q_words]
        answer = entity_dict[answer]
        re_document = []
        for u in new_document:
            u_words = ' '.join([entity_dict[w] if w in entity_dict else w for w in u])
            re_document.append(u_words)
        document = re_document
        question = ' '.join(q_words)

        questions.append(question)
        answers.append(answer)
        documents.append(document)

    logging.info('max_utterance_length %d' % (max_d_len))
    logging.info('max_query_length %d' % (max_q_len))
    logging.info('max_dialog_length %d' % (max_s_len))
    return (documents, questions, answers), max_d_len, max_q_len, max_s_len


def get_minibatches(n, minibatch_size, shuffle=False):
    idx_list = np.arange(0, n, minibatch_size)
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    for idx in idx_list:
        minibatches.append(np.arange(idx, min(idx + minibatch_size, n)))
    return minibatches


def get_dim(in_file):
    line = open(in_file).readline()
    return len(line.split()) - 1


def gen_embeddings(word_dict, dim, in_file=None):
    num_words = max(word_dict.values()) + 1
    embeddings = np.random.uniform(low=-0.01, high=0.01, size=(num_words, dim))
    logging.info('Embeddings: %d x %d' % (num_words, dim))

    if in_file is not None:
        logging.info('Loading embedding file: %s' % in_file)
        pre_trained = 0
        for line in open(in_file).readlines():
            sp = line.split()
            assert len(sp) == dim + 1
            if sp[0] in word_dict:
                pre_trained += 1
                embeddings[word_dict[sp[0]]] = [float(x) for x in sp[1:]]
        logging.info('Pre-trained: %d (%.2f%%)' %
                     (pre_trained, pre_trained * 100.0 / num_words))
    return embeddings


