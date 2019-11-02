import codecs

import numpy as np
import random
import tensorflow as tf
import os
from keras import backend as k
from optparse import OptionParser
import time
import sys
import json
import tools
from collections import Counter
from collections import OrderedDict
from nn_models import *
from keras.models import load_model
import logging
from keras.models import load_model

def gen_examples(token_inputs, seg_inputs, l, y, en_ms, en_ds, batch_size):
    minibatches = tools.get_minibatches(len(token_inputs), batch_size)
    all_ex = []
    for minibatch in minibatches:
        mb_token_inputs = [token_inputs[t] for t in minibatch]
        mb_seg_inputs = [seg_inputs[t] for t in minibatch]
        mb_l = l[minibatch]
        mb_y = [y[t] for t in minibatch]
        mb_en_ms = [en_ms[t] for t in minibatch]
        mb_en_ds = [en_ds[t] for t in minibatch]
        all_ex.append((mb_token_inputs, mb_seg_inputs, mb_l, mb_y, mb_en_ms, mb_en_ds))
    return all_ex


def pre_shuffle(token_inputs, seg_inputs, l, y, en_ms, en_ds):
    combine = list(zip(token_inputs, seg_inputs, l, y, en_ms, en_ds))
    np.random.shuffle(combine)
    token_inputs, seg_inputs, l, y, en_ms, en_ds = zip(*combine)
    return list(token_inputs), list(seg_inputs), np.array(l), list(y), list(en_ms), list(en_ds)


def accuracy_score(y_pred, y_true):
    assert len(y_pred) == len(y_true)

    if len(y_true) == 0:
        return 0.

    correctly = 0

    for pred, true in zip(y_pred, y_true):
        if pred == true:
            correctly += 1

    return float(correctly)


def eval_acc(any_model, all_examples):
    acc = 0
    n_examples = 0
    for token_inputs, seg_inputs, l, y, en_ms, en_ds in all_examples:
        # predictions = []
        # for token_input, seg_input, ll in zip(token_inputs, seg_inputs, l):
        #     inputs = [token_input, seg_input, l]
        #     prediction = any_model.predict_classes(inputs, np.array(ll))
        #     predictions.append(prediction[0])
        predictions = any_model.predict_classes([np.array(token_inputs)] + [np.array(seg_inputs)]+[np.array(en_ds)] + [np.array(en_ms)] + [np.array(l)],
                                                np.array(l))
        dev_pred = accuracy_score(predictions, y)
        acc += dev_pred
        n_examples += len(token_inputs)
    return acc * 100.0 / n_examples

def bert_fine_tuning(args):
    logger_exp.info('-' * 50)
    logger_exp.info('Load data files..')
    # get prune dictionaries
    redundent_1, redundent_2 = tools.prune_data(args.train_file)
    # load training data
    train_examples, max_d, max_q, max_s = tools.load_jsondata(args.train_file, redundent_1, redundent_2, args.stopwords)
    # load development data
    dev_examples, a, b, c = tools.load_jsondata(args.dev_file, redundent_1, redundent_2, args.stopwords)

    logger_exp.info('-' * 50)
    logger_exp.info('Build dictionary..')
    word_dict = tools.build_dict(train_examples[0], train_examples[1])
    # entity dictionary for entire dataset
    entity_markers = list(set([w for w in word_dict.keys() if w.startswith('@ent')] + train_examples[2]))
    entity_markers = ['<unk_entity>'] + entity_markers
    entity_dict = {w: index for (index, w) in enumerate(entity_markers)}
    logger_exp.info('Entity markers: %d' % len(entity_dict))
    num_labels = len(entity_dict)
    print(entity_dict)
    token_dict = {}
    with open(args.dict_path, 'r') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)


    logger_exp.info('-' * 50)
    logger_exp.info('Building Model..')
    if args.test_only:
        training = False
    else:
        training = True
    # build model
    if args.model_to_run == 'bert':
        bert_model = Bert_Model('Bert_Model',
                                num_labels,
                                config_path=args.config_path,
                                model_path=args.model_path,
                                learning_rate=args.learning_rate,
                                drop_out=args.drop_out, training=training)
    else:
        raise Exception('model `%s` not implemented' % args.model_to_run)
    if args.pre_trained is None:
        bert_model.set_bert_weights(args.config_path, args.model_path)
    if args.pre_trained is not None:
        bert_model.load_weights(args.pre_trained)

    logger_exp.info('Done.')

    logger_exp.info('-' * 50)
    logger_exp.info(args)

    logger_exp.info('-' * 50)
    logger_exp.info('Intial test..')
    # vectorize development data
    dev_token_inputs, dev_seg_inputs, dev_l, dev_y, dev_en_ms, dev_en_ds = tools.bert_vectorize(dev_examples,
                                                                                                token_dict,
                                                                                                entity_dict)
    # assert len(dev_token_inputs) == num_dev
    # assert len(dev_seg_inputs) == num_dev

    all_dev = gen_examples(dev_token_inputs, dev_seg_inputs, dev_l, dev_y, dev_en_ms, dev_en_ds, args.batch_size)
    dev_acc = eval_acc(bert_model, all_dev)
    logger_exp.info('Dev accuracy: %.2f %%' % dev_acc)
    best_acc = dev_acc
    if args.test_only:
        return
    bert_model.save_model(args.save_model)

    # Training
    logger_exp.info('-' * 50)
    logger_exp.info('Start training..')

    # vectorize training data
    train_token_inputs, train_seg_inputs, train_l, train_y, train_en_ms, train_en_ds = tools.bert_vectorize(train_examples,
                                                                                               token_dict,
                                                                                               entity_dict)
    num_train = len(dev_seg_inputs)
    num_dev = len(dev_seg_inputs)
    train_token_inputs, train_seg_inputs, train_l, train_y, train_en_ms, train_en_ds = pre_shuffle(train_token_inputs,
                                                                                                   train_seg_inputs,
                                                                                                   train_l,
                                                                                                   train_y,
                                                                                                   train_en_ms,
                                                                                                   train_en_ds)
    start_time = time.time()
    n_updates = 0
    all_train = gen_examples(train_token_inputs, train_seg_inputs, train_l, train_y, train_en_ms, train_en_ds, args.batch_size)

    for epoch in range(args.nb_epoch):
        np.random.shuffle(all_train)

        for idx, (mb_token_inputs, mb_seg_inputs, mb_l, mb_y, mb_en_ms, mb_en_ds) in enumerate(all_train):
            logger_exp.info('#Examples = %d' % (len(mb_token_inputs)))
            # rearrange each batch of dialogs
            hist = bert_model.fit([np.array(mb_token_inputs)] + [np.array(mb_seg_inputs)]+[np.array(mb_en_ds)]+[np.array(mb_en_ms)] + [np.array(mb_l)],
                                  np.array(mb_y)
                                  , batch_size=args.batch_size, verbose=0)
            logger_exp.info('Epoch = %d, iter = %d (max = %d), loss = %.2f, elapsed time = %.2f (s)' %
                            (epoch, idx, len(all_train), hist.history['loss'][0], time.time() - start_time))
            print('Epoch = %d, iter = %d (max = %d), loss = %.2f, elapsed time = %.2f (s)'
                  % (epoch, idx, len(all_train), hist.history['loss'][0], time.time() - start_time))
            n_updates += 1
            # evaluate every 100 batches
            if n_updates % 100 == 0:
                samples = sorted(np.random.choice(num_train, min(num_train, num_dev),
                                                  replace=False))
                sample_train = gen_examples([train_token_inputs[k] for k in samples],
                                            [train_seg_inputs[k] for k in samples],
                                            train_l[samples],
                                            [train_y[k] for k in samples],
                                            [train_en_ms[k] for k in samples],
                                            [train_en_ds[k] for k in samples],
                                            args.batch_size)
                train_acc = eval_acc(bert_model, sample_train)
                logger_exp.info('Train accuracy: %.2f %%' % train_acc)
                print('Train accuracy: %.2f %%' % train_acc)
                dev_acc = eval_acc(bert_model, all_dev)
                logger_exp.info('Dev accuracy: %.2f %%' % dev_acc)
                print('Dev accuracy: %.2f %%' % dev_acc)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    logger_exp.info('Best dev accuracy: epoch = %d, n_udpates = %d, acc = %.2f %%'
                                    % (epoch, n_updates, dev_acc))
                    print('Best dev accuracy: epoch = %d, n_udpates = %d, acc = %.2f %%'
                          % (epoch, n_updates, dev_acc))
                    bert_model.save_model(args.save_model)


if __name__ == '__main__':

    model_dict = ['bert']
    parser = OptionParser(usage="usage: %prog [options]")
    parser.add_option('--logging_to_file',
                      action='store',
                      dest='logging_to_file',
                      default=None,
                      help='logging to a file or stdout')
    parser.add_option('--model',
                      action='store',
                      dest='model_to_run',
                      default=None,
                      help='model to train (available: %s)' % ', '.join(model_dict))
    parser.add_option('--nb_epoch',
                      action='store',
                      dest='nb_epoch',
                      default=100,
                      help='number of epochs to train the model with')
    parser.add_option('--train_file',
                      action='store',
                      dest='train_file',
                      default=None,
                      help='train file')
    parser.add_option('--dev_file',
                      action='store',
                      dest='dev_file',
                      default=None,
                      help='dev file')
    parser.add_option('--save_model',
                      action='store',
                      dest='save_model',
                      default=None,
                      help='model to save')
    parser.add_option('--random_seed',
                      action='store',
                      dest='random_seed',
                      default=1234,
                      help='random seed')
    parser.add_option('--config_path',
                      action='store',
                      dest='config_path',
                      default=None,
                      help='bert config path')
    parser.add_option('--dict_path',
                      action='store',
                      dest='dict_path',
                      default=None,
                      help='bert dict path')
    parser.add_option('--model_path',
                      action='store',
                      dest='model_path',
                      default=None,
                      help='bert model_path path')
    parser.add_option('--stopwords',
                      action='store',
                      dest='stopwords',
                      default=None,
                      help='stopwords')
    parser.add_option('--test_only',
                      action='store',
                      dest='test_only',
                      default=False,
                      help='If just to test the model')
    parser.add_option('--pre_trained',
                      action='store',
                      dest='pre_trained',
                      default=None,
                      help='pre-trained model')
    parser.add_option('--batch_size',
                      action='store',
                      dest='batch_size',
                      default=6,
                      help='training and testing batch size')
    parser.add_option('--learning_rate',
                      action='store',
                      dest='learning_rate',
                      default=2e-5,
                      help='learning rate of the model')
    parser.add_option('--drop_out',
                      action='store',
                      dest='drop_out',
                      default=0.1,
                      help='learning rate of the model')
    parser.add_option('--gpu',
                      action='store',
                      dest='gpu',
                      default=None,
                      help='gpu to train')
    (options, args) = parser.parse_args()
    if options.gpu is None:
        raise ValueError('gpu is not specified.')
    os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    k.tensorflow_backend.set_session(tf.Session(config=config))
    fixed_seed_num = int(options.random_seed)
    options.nb_epoch = int(options.nb_epoch)
    options.batch_size = int(options.batch_size)
    options.learning_rate = float(options.learning_rate)
    np.random.seed(fixed_seed_num)
    random.seed(fixed_seed_num)
    tf.set_random_seed(fixed_seed_num)
    if options.train_file is None:
        raise ValueError('train_file is not specified.')
    if options.dev_file is None:
        raise ValueError('dev_file is not specified.')
    if options.stopwords is None:
        raise ValueError('stopwords are not specified.')
    if options.model_path is None:
        raise ValueError('bert model path are not specified.')
    if options.dict_path is None:
        raise ValueError('bert dict path are not specified.')
    if options.config_path is None:
        raise ValueError('bert config path are not specified.')
    FORMAT = '[%(levelname)-8s] [%(asctime)s] [%(name)-15s]: %(message)s'
    DATEFORMAT = '%Y-%m-%d %H:%M:%S'

    if options.logging_to_file:
        logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATEFORMAT,
                            filename=options.logging_to_file)
    else:
        logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATEFORMAT)

    logger_exp = logging.getLogger('experiments')

    if not options.logging_to_file:
        logger_exp.info('logging to stdout')

    if options.model_to_run not in model_dict:
        raise Exception('model `%s` not implemented' % options.model_to_run)

    bert_fine_tuning(options)
