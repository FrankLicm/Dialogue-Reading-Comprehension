from base import BaseModel
from keras.layers import Embedding, Input, GlobalMaxPooling1D, Layer
from keras.layers.merge import Concatenate, Multiply, dot, Dot
from keras.layers.core import Dense, Lambda, Reshape, Dropout
from keras.models import Model
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.optimizers import *
from keras.initializers import truncated_normal, zeros
from keras.layers.convolutional import *
import keras.backend as K
from keras.legacy import interfaces
from keras.optimizers import Optimizer
from keras import regularizers
from keras.layers.core import Reshape, Permute, Dense, Flatten, Lambda
from keras.activations import softmax
import numpy as np
import tensorflow as tf
from bert_loader_custom import load_trained_model_from_checkpoint
from keras_pos_embd import PositionEmbedding
from keras_layer_normalization import LayerNormalization
from keras_transformer import get_encoders
from keras_transformer import get_custom_objects as get_encoder_custom_objects
from keras_bert.layers import (get_inputs, get_embedding, TokenEmbedding, EmbeddingSimilarity, Masked, Extract)
import json
from keras.models import load_model
from keras_transformer import *
from keras_bert.layers import MaskedGlobalMaxPool1D
from keras_transformer.transformer import _wrap_layer
import keras
from keras.utils import multi_gpu_model


class CNN_LSTM_UA_DA_Model(BaseModel):
    def __init__(self, name, nb_classes, vocabulary_size, embedding_size, nb_utterance_token,
                 nb_query_token, nb_utterances, nb_filters_utterance=50, nb_filters_query=50,
                 learning_rate=0.001, dropout=0.2, nb_hidden_unit=32):

        self.nb_classes = nb_classes
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        # number of tokens per utterance
        self.nb_utterance_token = nb_utterance_token
        # number of tokens per query
        self.nb_query_token = nb_query_token
        # number of utterance per dialog
        self.nb_utterances = nb_utterances
        # number of filters in utterance convolution and query convolution 
        self.nb_filters_utterance = nb_filters_utterance
        self.nb_filters_query = nb_filters_query
        # hidden unit size of LSTM
        self.nb_hidden_unit = nb_hidden_unit
        self.learning_rate = learning_rate
        self.dropout = dropout

        self.embedding_layer_utterance = None
        self.embedding_layer_query = None
        self.embedding_set = False

        model = self.build_model()
        super(CNN_LSTM_UA_DA_Model, self).__init__(name, model)

    def masking_lambda(self, x):
        # masking out probabilities of entities that don't appear
        m_classes, m_masks = x[0], x[1]
        masked_classes = m_classes * m_masks
        masked_sum_ = K.sum(masked_classes, axis=1)
        masked_sum_ = K.expand_dims(masked_sum_, -1)
        masked_sum = K.repeat_elements(masked_sum_, self.nb_classes, axis=1)
        masked_classes = masked_classes / masked_sum
        masked_classes = K.clip(masked_classes, 1e-7, 1.0 - 1e-7)
        return masked_classes

    def crossatt(self, x):
        doc, query, doc_mask, q_mask = x[0], x[1], x[2], x[3]
        trans_doc = K.permute_dimensions(doc, (0, 2, 1))
        match_score = K.tanh(dot([query, trans_doc], (2, 1)))
        query_to_doc_att = K.softmax(K.sum(match_score, axis=1))
        doc_to_query_att = K.softmax(K.sum(match_score, axis=-1))

        alpha = query_to_doc_att * doc_mask
        a_sum = K.sum(alpha, axis=1)
        _a_sum = K.expand_dims(a_sum, -1)
        alpha = alpha / _a_sum

        beta = doc_to_query_att * q_mask
        b_sum = K.sum(beta, axis=1)
        _b_sum = K.expand_dims(b_sum, 1)
        beta = beta / _b_sum

        doc_vector = dot([trans_doc, alpha], (2, 1))
        trans_que = K.permute_dimensions(query, (0, 2, 1))
        que_vector = dot([trans_que, beta], (2, 1))
        final_hidden = K.concatenate([doc_vector, que_vector])
        return final_hidden

    def build_model(self):

        inputs = []
        # utterances 
        for i in range(self.nb_utterances):
            inputs.append(Input(shape=(self.nb_utterance_token,)))
        # similarity matrices 
        for i in range(self.nb_utterances):
            inputs.append(Input(shape=(self.nb_utterance_token, self.nb_query_token)))
        # query 
        inputs.append(Input(shape=(self.nb_query_token,)))
        # entity mask
        inputs.append(Input(shape=(self.nb_classes,)))
        # query token mask
        inputs.append(Input(shape=(self.nb_query_token,)))
        # dialog mask
        inputs.append(Input(shape=(self.nb_utterances,)))

        # embedding layer for utterances and query
        self.embedding_layer_utterance = Embedding(self.vocabulary_size, self.embedding_size)
        self.embedding_layer_query = Embedding(self.vocabulary_size, self.embedding_size,
                                               input_length=self.nb_query_token, mask_zero=True)
        # utternace level attention matrix
        attn = DocAttentionMap((self.nb_utterance_token, self.embedding_size))
        # 3-D embedding for utterances
        embedding_utterances = []
        for i in range(self.nb_utterances):
            embedding_utter = Reshape((self.nb_utterance_token, self.embedding_size, 1))(
                self.embedding_layer_utterance(inputs[i]))
            doc_att_map = Reshape((self.nb_utterance_token, self.embedding_size, 1))(
                attn(inputs[i + self.nb_utterances]))
            embedding_utterances.append(Concatenate()([embedding_utter, doc_att_map]))

        # convolution embedding input for query
        conv_embedding_query = Reshape((self.nb_query_token, self.embedding_size, 1))(
            self.embedding_layer_utterance(inputs[-4]))
        # LSTM embedding input for query
        embedding_query = self.embedding_layer_query(inputs[-4])
        # convolution output for query
        conv_q = Reshape((self.nb_query_token, self.nb_filters_query))(
            Convolution2D(self.nb_filters_query, (1, self.embedding_size), activation='relu')(conv_embedding_query))
        # utterance embeddings
        scene = []
        for i in range(self.nb_utterances):
            utter = []
            for j in range(2, 6):
                conv_u = Convolution2D(self.nb_filters_utterance, (j, self.embedding_size), activation='relu')(
                    embedding_utterances[i])
                pool_u = Reshape((self.nb_filters_utterance,))(
                    MaxPooling2D(pool_size=(self.nb_utterance_token - j + 1, 1))(conv_u))
                utter.append(pool_u)
            scene.append(Reshape((self.nb_filters_utterance * 4, 1))(Concatenate()(utter)))
        # dialog matrix
        scene = Permute((2, 1))(Concatenate()(scene))
        # convolution output of dialog matrix 
        reshape_scene = Reshape((self.nb_utterances, self.nb_filters_utterance * 4, 1))(scene)
        single = Convolution2D(self.nb_filters_utterance, (1, self.nb_filters_utterance * 4), activation='relu')(
            reshape_scene)
        single = Reshape((self.nb_utterances, self.nb_filters_utterance))(single)

        # context embedding for both dialog and query
        d_rnn_layer = LSTM(self.nb_hidden_unit, activation='tanh', dropout=self.dropout)
        q_rnn_layer = LSTM(self.nb_hidden_unit, activation='tanh', dropout=self.dropout)
        bi_d_rnn = Bidirectional(d_rnn_layer, merge_mode='concat')(scene)
        bi_q_rnn = Bidirectional(q_rnn_layer, merge_mode='concat')(embedding_query)
        # dialog level attention vector
        att_vector = Lambda(self.crossatt, output_shape=(self.nb_filters_utterance * 2,))(
            [single, conv_q, inputs[-1], inputs[-2]])

        merged_vectors = Concatenate()([bi_d_rnn, bi_q_rnn, att_vector])
        classes = Dense(units=self.nb_classes, activation='softmax')(merged_vectors)
        # masking 
        classes_normalized = Lambda(self.masking_lambda, output_shape=(self.nb_classes,))([classes, inputs[-3]])
        model = Model(inputs=inputs, outputs=classes_normalized)
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt)
        return model

    def load_embedding(self, embedding):
        if self.model is None:
            raise Exception('model has not been built')

        self.embedding_layer_query.set_weights(embedding)
        self.embedding_layer_utterance.set_weights(embedding)
        self.embedding_set = True

    def fit(self, x, y, *args, **kwargs):
        if self.embedding_set is False:
            raise Exception('embedding has not bet set')
        hist = self.model.fit(x, y, *args, **kwargs)
        return hist

    def predict_classes(self, x, y_masks):
        predictions = self.model.predict(x)
        predictions_masked = predictions * y_masks
        classes = [np.argmax(i) for i in predictions_masked]
        return classes


class CNN_LSTM_UA_Model(BaseModel):
    def __init__(self, name, nb_classes, vocabulary_size, embedding_size, nb_utterance_token,
                 nb_query_token, nb_utterances, nb_filters_utterance=50, nb_filters_query=50,
                 learning_rate=0.001, dropout=0.2, nb_hidden_unit=32):

        self.nb_classes = nb_classes
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        # number of tokens per utterance
        self.nb_utterance_token = nb_utterance_token
        # number of tokens per query
        self.nb_query_token = nb_query_token
        # number of utterance per dialog
        self.nb_utterances = nb_utterances
        # number of filters in utterance convolution and query convolution 
        self.nb_filters_utterance = nb_filters_utterance
        self.nb_filters_query = nb_filters_query
        # hidden unit size of LSTM
        self.nb_hidden_unit = nb_hidden_unit
        self.learning_rate = learning_rate
        self.dropout = dropout

        self.embedding_layer_utterance = None
        self.embedding_layer_query = None
        self.embedding_set = False

        model = self.build_model()
        super(CNN_LSTM_UA_Model, self).__init__(name, model)

    def masking_lambda(self, x):
        # masking out probabilities of entities that don't appear
        m_classes, m_masks = x[0], x[1]
        masked_classes = m_classes * m_masks
        masked_sum_ = K.sum(masked_classes, axis=1)
        masked_sum_ = K.expand_dims(masked_sum_, -1)
        masked_sum = K.repeat_elements(masked_sum_, self.nb_classes, axis=1)
        masked_classes = masked_classes / masked_sum
        masked_classes = K.clip(masked_classes, 1e-7, 1.0 - 1e-7)
        return masked_classes

    def build_model(self):

        inputs = []
        # utterances 
        for i in range(self.nb_utterances):
            inputs.append(Input(shape=(self.nb_utterance_token,)))
        # similarity matrices 
        for i in range(self.nb_utterances):
            inputs.append(Input(shape=(self.nb_utterance_token, self.nb_query_token)))
        # query 
        inputs.append(Input(shape=(self.nb_query_token,)))
        # entity mask
        inputs.append(Input(shape=(self.nb_classes,)))
        # query token mask
        inputs.append(Input(shape=(self.nb_query_token,)))
        # dialog mask
        inputs.append(Input(shape=(self.nb_utterances,)))

        # embedding layer for utterances and query
        self.embedding_layer_utterance = Embedding(self.vocabulary_size, self.embedding_size)
        self.embedding_layer_query = Embedding(self.vocabulary_size, self.embedding_size,
                                               input_length=self.nb_query_token, mask_zero=True)
        # utternace level attention matrix
        attn = DocAttentionMap((self.nb_utterance_token, self.embedding_size))
        # 3-D embedding for utterances
        embedding_utterances = []
        for i in range(self.nb_utterances):
            embedding_utter = Reshape((self.nb_utterance_token, self.embedding_size, 1))(
                self.embedding_layer_utterance(inputs[i]))
            doc_att_map = Reshape((self.nb_utterance_token, self.embedding_size, 1))(
                attn(inputs[i + self.nb_utterances]))
            embedding_utterances.append(Concatenate()([embedding_utter, doc_att_map]))

        # LSTM embedding input for query
        embedding_query = self.embedding_layer_query(inputs[-4])
        # utterance embeddings
        scene = []
        for i in range(self.nb_utterances):
            utter = []
            for j in range(2, 6):
                conv_u = Convolution2D(self.nb_filters_utterance, (j, self.embedding_size), activation='relu')(
                    embedding_utterances[i])
                pool_u = Reshape((self.nb_filters_utterance,))(
                    MaxPooling2D(pool_size=(self.nb_utterance_token - j + 1, 1))(conv_u))
                utter.append(pool_u)
            scene.append(Reshape((self.nb_filters_utterance * 4, 1))(Concatenate()(utter)))
        # dialog matrix
        scene = Permute((2, 1))(Concatenate()(scene))

        # context embedding for both dialog and query
        d_rnn_layer = LSTM(self.nb_hidden_unit, activation='tanh', dropout=self.dropout)
        q_rnn_layer = LSTM(self.nb_hidden_unit, activation='tanh', dropout=self.dropout)
        bi_d_rnn = Bidirectional(d_rnn_layer, merge_mode='concat')(scene)
        bi_q_rnn = Bidirectional(q_rnn_layer, merge_mode='concat')(embedding_query)

        merged_vectors = Concatenate()([bi_d_rnn, bi_q_rnn])
        classes = Dense(units=self.nb_classes, activation='softmax')(merged_vectors)
        # masking 
        classes_normalized = Lambda(self.masking_lambda, output_shape=(self.nb_classes,))([classes, inputs[-3]])
        model = Model(inputs=inputs, outputs=classes_normalized)
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt)
        return model

    def load_embedding(self, embedding):
        if self.model is None:
            raise Exception('model has not been built')

        self.embedding_layer_query.set_weights(embedding)
        self.embedding_layer_utterance.set_weights(embedding)
        self.embedding_set = True

    def fit(self, x, y, *args, **kwargs):
        if self.embedding_set is False:
            raise Exception('embedding has not bet set')
        hist = self.model.fit(x, y, *args, **kwargs)
        return hist

    def predict_classes(self, x, y_masks):
        predictions = self.model.predict(x)
        predictions_masked = predictions * y_masks
        classes = [np.argmax(i) for i in predictions_masked]
        return classes


class CNN_LSTM_DA_Model(BaseModel):
    def __init__(self, name, nb_classes, vocabulary_size, embedding_size, nb_utterance_token,
                 nb_query_token, nb_utterances, nb_filters_utterance=50, nb_filters_query=50,
                 learning_rate=0.001, dropout=0.2, nb_hidden_unit=32):

        self.nb_classes = nb_classes
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        # number of tokens per utterance
        self.nb_utterance_token = nb_utterance_token
        # number of tokens per query
        self.nb_query_token = nb_query_token
        # number of utterance per dialog
        self.nb_utterances = nb_utterances
        # number of filters in utterance convolution and query convolution 
        self.nb_filters_utterance = nb_filters_utterance
        self.nb_filters_query = nb_filters_query
        # hidden unit size of LSTM
        self.nb_hidden_unit = nb_hidden_unit
        self.learning_rate = learning_rate
        self.dropout = dropout

        self.embedding_layer_utterance = None
        self.embedding_layer_query = None
        self.embedding_set = False

        model = self.build_model()
        super(CNN_LSTM_DA_Model, self).__init__(name, model)

    def masking_lambda(self, x):
        # masking out probabilities of entities that don't appear
        m_classes, m_masks = x[0], x[1]
        masked_classes = m_classes * m_masks
        masked_sum_ = K.sum(masked_classes, axis=1)
        masked_sum_ = K.expand_dims(masked_sum_, -1)
        masked_sum = K.repeat_elements(masked_sum_, self.nb_classes, axis=1)
        masked_classes = masked_classes / masked_sum
        masked_classes = K.clip(masked_classes, 1e-7, 1.0 - 1e-7)
        return masked_classes

    def crossatt(self, x):
        doc, query, doc_mask, q_mask = x[0], x[1], x[2], x[3]
        trans_doc = K.permute_dimensions(doc, (0, 2, 1))
        match_score = K.tanh(dot([query, trans_doc], (2, 1)))
        query_to_doc_att = K.softmax(K.sum(match_score, axis=1))
        doc_to_query_att = K.softmax(K.sum(match_score, axis=-1))

        alpha = query_to_doc_att * doc_mask
        a_sum = K.sum(alpha, axis=1)
        _a_sum = K.expand_dims(a_sum, -1)
        alpha = alpha / _a_sum

        beta = doc_to_query_att * q_mask
        b_sum = K.sum(beta, axis=1)
        _b_sum = K.expand_dims(b_sum, 1)
        beta = beta / _b_sum

        doc_vector = dot([trans_doc, alpha], (2, 1))
        trans_que = K.permute_dimensions(query, (0, 2, 1))
        que_vector = dot([trans_que, beta], (2, 1))
        final_hidden = K.concatenate([doc_vector, que_vector])
        return final_hidden

    def build_model(self):

        inputs = []
        # utterances 
        for i in range(self.nb_utterances):
            inputs.append(Input(shape=(self.nb_utterance_token,)))
        # query 
        inputs.append(Input(shape=(self.nb_query_token,)))
        # entity mask
        inputs.append(Input(shape=(self.nb_classes,)))
        # query token mask
        inputs.append(Input(shape=(self.nb_query_token,)))
        # dialog mask
        inputs.append(Input(shape=(self.nb_utterances,)))

        # embedding layer for utterances and query
        self.embedding_layer_utterance = Embedding(self.vocabulary_size, self.embedding_size)
        self.embedding_layer_query = Embedding(self.vocabulary_size, self.embedding_size,
                                               input_length=self.nb_query_token, mask_zero=True)

        # 2-D embedding for utterances
        embedding_utterances = []
        for i in range(self.nb_utterances):
            embedding_utter = Reshape((self.nb_utterance_token, self.embedding_size, 1))(
                self.embedding_layer_utterance(inputs[i]))
            embedding_utterances.append(embedding_utter)

        # convolution embedding input for query
        conv_embedding_query = Reshape((self.nb_query_token, self.embedding_size, 1))(
            self.embedding_layer_utterance(inputs[-4]))
        # LSTM embedding input for query
        embedding_query = self.embedding_layer_query(inputs[-4])
        # convolution output for query
        conv_q = Reshape((self.nb_query_token, self.nb_filters_query))(
            Convolution2D(self.nb_filters_query, (1, self.embedding_size), activation='relu')(conv_embedding_query))
        # utterance embeddings
        scene = []
        for i in range(self.nb_utterances):
            utter = []
            for j in range(2, 6):
                conv_u = Convolution2D(self.nb_filters_utterance, (j, self.embedding_size), activation='relu')(
                    embedding_utterances[i])
                pool_u = Reshape((self.nb_filters_utterance,))(
                    MaxPooling2D(pool_size=(self.nb_utterance_token - j + 1, 1))(conv_u))
                utter.append(pool_u)
            scene.append(Reshape((self.nb_filters_utterance * 4, 1))(Concatenate()(utter)))
        # dialog matrix
        scene = Permute((2, 1))(Concatenate()(scene))
        # convolution output of dialog matrix 
        reshape_scene = Reshape((self.nb_utterances, self.nb_filters_utterance * 4, 1))(scene)
        single = Convolution2D(self.nb_filters_utterance, (1, self.nb_filters_utterance * 4), activation='relu')(
            reshape_scene)
        single = Reshape((self.nb_utterances, self.nb_filters_utterance))(single)

        # context embedding for both dialog and query
        d_rnn_layer = LSTM(self.nb_hidden_unit, activation='tanh', dropout=self.dropout)
        q_rnn_layer = LSTM(self.nb_hidden_unit, activation='tanh', dropout=self.dropout)
        bi_d_rnn = Bidirectional(d_rnn_layer, merge_mode='concat')(scene)
        bi_q_rnn = Bidirectional(q_rnn_layer, merge_mode='concat')(embedding_query)
        # dialog level attention vector
        att_vector = Lambda(self.crossatt, output_shape=(self.nb_filters_utterance * 2,))(
            [single, conv_q, inputs[-1], inputs[-2]])

        merged_vectors = Concatenate()([bi_d_rnn, bi_q_rnn, att_vector])
        classes = Dense(units=self.nb_classes, activation='softmax')(merged_vectors)
        # masking 
        classes_normalized = Lambda(self.masking_lambda, output_shape=(self.nb_classes,))([classes, inputs[-3]])
        model = Model(inputs=inputs, outputs=classes_normalized)
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt)
        return model

    def load_embedding(self, embedding):
        if self.model is None:
            raise Exception('model has not been built')

        self.embedding_layer_query.set_weights(embedding)
        self.embedding_layer_utterance.set_weights(embedding)
        self.embedding_set = True

    def fit(self, x, y, *args, **kwargs):
        if self.embedding_set is False:
            raise Exception('embedding has not bet set')
        hist = self.model.fit(x, y, *args, **kwargs)
        return hist

    def predict_classes(self, x, y_masks):
        predictions = self.model.predict(x)
        predictions_masked = predictions * y_masks
        classes = [np.argmax(i) for i in predictions_masked]
        return classes


class CNN_LSTM_Model(BaseModel):
    def __init__(self, name, nb_classes, vocabulary_size, embedding_size, nb_utterance_token,
                 nb_query_token, nb_utterances, nb_filters_utterance=50, nb_filters_query=50,
                 learning_rate=0.001, dropout=0.2, nb_hidden_unit=32):

        self.nb_classes = nb_classes
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        # number of tokens per utterance
        self.nb_utterance_token = nb_utterance_token
        # number of tokens per query
        self.nb_query_token = nb_query_token
        # number of utterance per dialog
        self.nb_utterances = nb_utterances
        # number of filters in utterance convolution and query convolution 
        self.nb_filters_utterance = nb_filters_utterance
        self.nb_filters_query = nb_filters_query
        # hidden unit size of LSTM
        self.nb_hidden_unit = nb_hidden_unit
        self.learning_rate = learning_rate
        self.dropout = dropout

        self.embedding_layer_utterance = None
        self.embedding_layer_query = None
        self.embedding_set = False

        model = self.build_model()
        super(CNN_LSTM_Model, self).__init__(name, model)

    def masking_lambda(self, x):
        # masking out probabilities of entities that don't appear
        m_classes, m_masks = x[0], x[1]
        masked_classes = m_classes * m_masks
        masked_sum_ = K.sum(masked_classes, axis=1)
        masked_sum_ = K.expand_dims(masked_sum_, -1)
        masked_sum = K.repeat_elements(masked_sum_, self.nb_classes, axis=1)
        masked_classes = masked_classes / masked_sum
        masked_classes = K.clip(masked_classes, 1e-7, 1.0 - 1e-7)
        return masked_classes

    def build_model(self):

        inputs = []
        # utterances 
        for i in range(self.nb_utterances):
            inputs.append(Input(shape=(self.nb_utterance_token,)))
        # query 
        inputs.append(Input(shape=(self.nb_query_token,)))
        # entity mask
        inputs.append(Input(shape=(self.nb_classes,)))
        # query token mask
        inputs.append(Input(shape=(self.nb_query_token,)))
        # dialog mask
        inputs.append(Input(shape=(self.nb_utterances,)))

        # embedding layer for utterances and query
        self.embedding_layer_utterance = Embedding(self.vocabulary_size, self.embedding_size)
        self.embedding_layer_query = Embedding(self.vocabulary_size, self.embedding_size,
                                               input_length=self.nb_query_token, mask_zero=True)

        # 2-D embedding for utterances
        embedding_utterances = []
        for i in range(self.nb_utterances):
            embedding_utter = Reshape((self.nb_utterance_token, self.embedding_size, 1))(
                self.embedding_layer_utterance(inputs[i]))
            embedding_utterances.append(embedding_utter)

        # LSTM embedding input for query
        embedding_query = self.embedding_layer_query(inputs[-4])
        # utterance embeddings
        scene = []
        for i in range(self.nb_utterances):
            utter = []
            for j in range(2, 6):
                conv_u = Convolution2D(self.nb_filters_utterance, (j, self.embedding_size), activation='relu')(
                    embedding_utterances[i])
                pool_u = Reshape((self.nb_filters_utterance,))(
                    MaxPooling2D(pool_size=(self.nb_utterance_token - j + 1, 1))(conv_u))
                utter.append(pool_u)
            scene.append(Reshape((self.nb_filters_utterance * 4, 1))(Concatenate()(utter)))
        # dialog matrix
        scene = Permute((2, 1))(Concatenate()(scene))

        # context embedding for both dialog and query
        d_rnn_layer = LSTM(self.nb_hidden_unit, activation='tanh', dropout=self.dropout)
        q_rnn_layer = LSTM(self.nb_hidden_unit, activation='tanh', dropout=self.dropout)
        bi_d_rnn = Bidirectional(d_rnn_layer, merge_mode='concat')(scene)
        bi_q_rnn = Bidirectional(q_rnn_layer, merge_mode='concat')(embedding_query)

        merged_vectors = Concatenate()([bi_d_rnn, bi_q_rnn])
        classes = Dense(units=self.nb_classes, activation='softmax')(merged_vectors)
        # masking 
        classes_normalized = Lambda(self.masking_lambda, output_shape=(self.nb_classes,))([classes, inputs[-3]])
        model = Model(inputs=inputs, outputs=classes_normalized)
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt)
        return model

    def load_embedding(self, embedding):
        if self.model is None:
            raise Exception('model has not been built')

        self.embedding_layer_query.set_weights(embedding)
        self.embedding_layer_utterance.set_weights(embedding)
        self.embedding_set = True

    def fit(self, x, y, *args, **kwargs):
        if self.embedding_set is False:
            raise Exception('embedding has not bet set')
        hist = self.model.fit(x, y, *args, **kwargs)
        return hist

    def predict_classes(self, x, y_masks):
        predictions = self.model.predict(x)
        predictions_masked = predictions * y_masks
        classes = [np.argmax(i) for i in predictions_masked]
        return classes


class LSTM_Model(BaseModel):
    def __init__(self, name, nb_classes, vocabulary_size, embedding_size, nb_utterance_token,
                 nb_query_token, nb_utterances, nb_filters_utterance=50, nb_filters_query=50,
                 learning_rate=0.001, dropout=0.2, nb_hidden_unit=32):

        self.nb_classes = nb_classes
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        # number of tokens per utterance
        self.nb_utterance_token = nb_utterance_token
        # number of tokens per query
        self.nb_query_token = nb_query_token
        # number of utterance per dialog
        self.nb_utterances = nb_utterances
        # number of filters in utterance convolution and query convolution
        self.nb_filters_utterance = nb_filters_utterance
        self.nb_filters_query = nb_filters_query
        # hidden unit size of LSTM
        self.nb_hidden_unit = nb_hidden_unit
        self.learning_rate = learning_rate
        self.dropout = dropout

        self.embedding_layer_utterance = None
        self.embedding_layer_query = None
        self.embedding_set = False

        model = self.build_model()
        super(LSTM_Model, self).__init__(name, model)

    def masking_lambda(self, x):
        # masking out probabilities of entities that don't appear
        m_classes, m_masks = x[0], x[1]
        masked_classes = m_classes * m_masks
        masked_sum_ = K.sum(masked_classes, axis=1)
        masked_sum_ = K.expand_dims(masked_sum_, -1)
        masked_sum = K.repeat_elements(masked_sum_, self.nb_classes, axis=1)
        masked_classes = masked_classes / masked_sum
        masked_classes = K.clip(masked_classes, 1e-7, 1.0 - 1e-7)
        return masked_classes

    def build_model(self):

        inputs = []
        # utterances
        for i in range(self.nb_utterances):
            inputs.append(Input(shape=(self.nb_utterance_token,)))
        # query
        inputs.append(Input(shape=(self.nb_query_token,)))
        # entity mask
        inputs.append(Input(shape=(self.nb_classes,)))
        # query token mask
        inputs.append(Input(shape=(self.nb_query_token,)))
        # dialog mask
        inputs.append(Input(shape=(self.nb_utterances,)))

        # embedding layer for utterances and query
        self.embedding_layer_utterance = Embedding(self.vocabulary_size, self.embedding_size)
        self.embedding_layer_query = Embedding(self.vocabulary_size, self.embedding_size,
                                               input_length=self.nb_query_token, mask_zero=True)

        # 2-D embedding for utterances
        embedding_utterances = []
        for i in range(self.nb_utterances):
            embedding_utter = (self.embedding_layer_utterance(inputs[i]))
            embedding_utterances.append(embedding_utter)

        # LSTM embedding input for query
        embedding_query = self.embedding_layer_query(inputs[-4])
        # utterance embeddings
        scene = []
        for i in range(self.nb_utterances):
            utter = []
            for j in range(2, 6):
                utter.append(embedding_utterances[i])
            scene.append((Concatenate()(utter)))
        # dialog matrix
        scene = (Concatenate()(scene))

        # context embedding for both dialog and query
        d_rnn_layer = LSTM(self.nb_hidden_unit, activation='tanh', dropout=self.dropout)
        q_rnn_layer = LSTM(self.nb_hidden_unit, activation='tanh', dropout=self.dropout)
        bi_d_rnn = Bidirectional(d_rnn_layer, merge_mode='concat')(scene)
        bi_q_rnn = Bidirectional(q_rnn_layer, merge_mode='concat')(embedding_query)
        merged_vectors = Concatenate()([bi_d_rnn, bi_q_rnn])
        classes = Dense(units=self.nb_classes, activation='softmax')(merged_vectors)
        # masking
        classes_normalized = Lambda(self.masking_lambda, output_shape=(self.nb_classes,))([classes, inputs[-3]])
        model = Model(inputs=inputs, outputs=classes_normalized)
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt)
        return model

    def load_embedding(self, embedding):
        if self.model is None:
            raise Exception('model has not been built')

        self.embedding_layer_query.set_weights(embedding)
        self.embedding_layer_utterance.set_weights(embedding)
        self.embedding_set = True

    def fit(self, x, y, *args, **kwargs):
        hist = self.model.fit(x, y, *args, **kwargs)
        return hist

    def predict_classes(self, x, y_masks):
        predictions = self.model.predict(x)
        predictions_masked = predictions * y_masks
        classes = [np.argmax(i) for i in predictions_masked]
        return classes


def gelu(x):
    return 0.5 * x * (1.0 + tf.erf(x / tf.sqrt(2.0)))


class Bert_Model(BaseModel):
    def __init__(self, name, nb_classes, config_path, model_path, learning_rate=2e-5, drop_out=0.1, training=False):

        self.nb_classes = nb_classes
        self.learning_rate = learning_rate
        self.config_path = config_path
        self.model_path = model_path
        self.drop_out = drop_out
        model = self.load_trained_model_from_checkpoint(self.config_path, self.model_path, training=training)
        print("build model success")
        super(Bert_Model, self).__init__(name, model)

    def masking_lambda(self, x):
        # masking out probabilities of entities that don't appear
        m_classes, m_masks = x[0], x[1]
        masked_classes = m_classes * m_masks
        masked_sum_ = K.sum(masked_classes, axis=1)
        masked_sum_ = K.expand_dims(masked_sum_, -1)
        masked_sum = K.repeat_elements(masked_sum_, self.nb_classes, axis=1)
        masked_classes = masked_classes / masked_sum
        masked_classes = K.clip(masked_classes, 1e-7, 1.0 - 1e-7)
        return masked_classes

    def fit(self, x, y, *args, **kwargs):
        hist = self.model.fit(x, y, *args, **kwargs)
        return hist

    def predict_classes(self, x, y_masks):
        predictions = self.model.predict(x)
        predictions_masked = predictions * y_masks
        classes = [np.argmax(i) for i in predictions_masked]
        return classes

    def get_model(self,
                  token_num,
                  pos_num=512,
                  seq_len=512,
                  embed_dim=768,
                  transformer_num=12,
                  head_num=12,
                  feed_forward_dim=3072,
                  dropout_rate=0.0,
                  attention_activation=None,
                  feed_forward_activation=gelu,
                  training=False):
        inputs = get_inputs(seq_len=seq_len)
        model_inputs = inputs[:2]
        model_inputs.append(Input(shape=(seq_len, self.nb_classes,)))
        model_inputs.append(Input(shape=(seq_len, embed_dim,)))
        model_inputs.append(Input(shape=(self.nb_classes,)))
        embed_layer, embed_weights = get_embedding(
            inputs,
            token_num=token_num,
            embed_dim=embed_dim,
            pos_num=pos_num,
            dropout_rate=dropout_rate,
            trainable=training,
        )
        transformed = embed_layer
        transformed = get_encoders(
            encoder_num=transformer_num,
            input_layer=transformed,
            head_num=head_num,
            hidden_dim=feed_forward_dim,
            attention_activation=attention_activation,
            feed_forward_activation=feed_forward_activation,
            dropout_rate=dropout_rate,
            trainable=training,
        )

        first_token_tensor = Lambda(lambda y: K.squeeze(y[:, 0:1, :], axis=1))(transformed)
        classes = Dense(units=self.nb_classes,
                        use_bias=True,
                        kernel_initializer=truncated_normal(stddev=0.02),
                        activation='softmax')(first_token_tensor)
        classes_normalized = Lambda(self.masking_lambda, output_shape=(self.nb_classes,))(
            [classes, model_inputs[-1]])
        model = Model(inputs=model_inputs, outputs=classes_normalized)
        opt = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-6, decay=0.01)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt)
        return model

    def checkpoint_loader(self, checkpoint_file):
        def _loader(name):
            return tf.train.load_variable(checkpoint_file, name)

        return _loader

    def load_trained_model_from_checkpoint(self, config_file,
                                           checkpoint_file,
                                           training=False,
                                           seq_len=None):
        with open(config_file, 'r') as reader:
            config = json.loads(reader.read())
        if seq_len is None:
            seq_len = config['max_position_embeddings']
        else:
            seq_len = min(seq_len, config['max_position_embeddings'])

        model = self.get_model(
            token_num=config['vocab_size'],
            pos_num=seq_len,
            seq_len=seq_len,
            embed_dim=config['hidden_size'],
            transformer_num=config['num_hidden_layers'],
            head_num=config['num_attention_heads'],
            feed_forward_dim=config['intermediate_size'],
            training=training,
        )
        return model

    def set_bert_weights(self, config_file, checkpoint_file, seq_len=512):
        with open(config_file, 'r') as reader:
            config = json.loads(reader.read())
        loader = self.checkpoint_loader(checkpoint_file)
        self.model.get_layer(name='Embedding-Token').set_weights([
            loader('bert/embeddings/word_embeddings'),
        ])
        self.model.get_layer(name='Embedding-Position').set_weights([
            loader('bert/embeddings/position_embeddings')[:seq_len, :],
        ])
        self.model.get_layer(name='Embedding-Segment').set_weights([
            loader('bert/embeddings/token_type_embeddings'),
        ])
        self.model.get_layer(name='Embedding-Norm').set_weights([
            loader('bert/embeddings/LayerNorm/gamma'),
            loader('bert/embeddings/LayerNorm/beta'),
        ])
        for i in range(config['num_hidden_layers']):
            self.model.get_layer(name='Encoder-%d-MultiHeadSelfAttention' % (i + 1)).set_weights([
                loader('bert/encoder/layer_%d/attention/self/query/kernel' % i),
                loader('bert/encoder/layer_%d/attention/self/query/bias' % i),
                loader('bert/encoder/layer_%d/attention/self/key/kernel' % i),
                loader('bert/encoder/layer_%d/attention/self/key/bias' % i),
                loader('bert/encoder/layer_%d/attention/self/value/kernel' % i),
                loader('bert/encoder/layer_%d/attention/self/value/bias' % i),
                loader('bert/encoder/layer_%d/attention/output/dense/kernel' % i),
                loader('bert/encoder/layer_%d/attention/output/dense/bias' % i),
            ])
            self.model.get_layer(name='Encoder-%d-MultiHeadSelfAttention-Norm' % (i + 1)).set_weights([
                loader('bert/encoder/layer_%d/attention/output/LayerNorm/gamma' % i),
                loader('bert/encoder/layer_%d/attention/output/LayerNorm/beta' % i),
            ])
            self.model.get_layer(name='Encoder-%d-MultiHeadSelfAttention-Norm' % (i + 1)).set_weights([
                loader('bert/encoder/layer_%d/attention/output/LayerNorm/gamma' % i),
                loader('bert/encoder/layer_%d/attention/output/LayerNorm/beta' % i),
            ])
            self.model.get_layer(name='Encoder-%d-FeedForward' % (i + 1)).set_weights([
                loader('bert/encoder/layer_%d/intermediate/dense/kernel' % i),
                loader('bert/encoder/layer_%d/intermediate/dense/bias' % i),
                loader('bert/encoder/layer_%d/output/dense/kernel' % i),
                loader('bert/encoder/layer_%d/output/dense/bias' % i),
            ])
            self.model.get_layer(name='Encoder-%d-FeedForward-Norm' % (i + 1)).set_weights([
                loader('bert/encoder/layer_%d/output/LayerNorm/gamma' % i),
                loader('bert/encoder/layer_%d/output/LayerNorm/beta' % i),
            ])


class DocAttentionMap(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.U = None
        super(DocAttentionMap, self).__init__(**kwargs)

    def get_config(self):
        base_config = super(DocAttentionMap, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    def build(self, input_shape):
        # print input_shapexsss
        self.U = self.add_weight(name='kernel',
                                 shape=(input_shape[-1], self.output_dim[1]),
                                 initializer='uniform',
                                 trainable=True)
        super(DocAttentionMap, self).build(input_shape)

    def call(self, x, **kwargs):
        # print 'x (Q): %s' % str(x._keras_shape)
        # print 'U    : %s' % str(self.U._keras_shape)
        xU = K.tanh(K.dot(x, self.U))
        return xU

    def compute_output_shape(self, input_shape):
        return None, self.output_dim[0], self.output_dim[1]

