import tensorflow as tf
import os
import sys
sys.path.append('./')
import tqdm
import time
from utils import *
import numpy as np
import logging
import tensorflow as tf
from model.utils.bimpm import match_utils, layer_utils
from model.model_template import ModelTemplate
path=os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0,path)
base_path=os.path.split(os.path.split(os.path.split(os.path.split(os.path.realpath(__file__))[0])[0])[0])[0]
gpu_id=1
os.environ["CUDA_VISIBLE_DEVICES"] = "%s"%gpu_id

class Bimpm(ModelTemplate):

    def __init__(self,args,scope):
        super(Bimpm, self).__init__(args,scope)
        self.scope=scope
        self.args=args
        self.saver=None
        self.sess=None
        self.args.aggregation_layer_num=1
        self.args.context_layer_num=1
        self.args.context_lstm_dim=100
        self.args.aggregation_lstm_dim=100

        self.args.with_highway=True
        self.args.with_match_highway=True
        self.args.with_aggregation_highway=True

        self.args.att_dim=50
        self.args.att_type="symmetric"

        self.args.use_cudnn=False

        self.args.with_full_match=True,
        self.args.with_maxpool_match=True
        self.args.with_attentive_match=True
        self.args.with_max_attentive_match=True

        self.args.dropout_rate=0.1
        self.args.with_cosine=True
        self.args.with_mp_cosine=True
        self.args.cosine_MP_dim=5


    def build_placeholder(self,):
        with tf.variable_scope(name_or_scope=self.scope):
            with tf.device('/device:GPU:%s'%gpu_id):

                self.sent1_word = tf.placeholder(shape=(None, self.args.seq_len), dtype=tf.int32,name='sent1_word')
                self.sent1_len = tf.placeholder(shape=(None,), dtype=tf.int32,name='sent1_len')
                self.sent2_word = tf.placeholder(shape=(None, self.args.seq_len), dtype=tf.int32,name='sent2_word')
                self.sent2_len = tf.placeholder(shape=(None,), dtype=tf.int32,name='sent2_len')
                self.dropout=tf.placeholder(dtype=tf.float32)
                self.target = tf.placeholder(shape=(None), dtype=tf.int32,name='target')
                self.sent1_token_mask = tf.cast(self.sent1_word, tf.bool)
                self.sent2_token_mask = tf.cast(self.sent2_word, tf.bool)

    def build_emb(self,input,reuse=False,*args,**kwargs):

        with tf.variable_scope(name_or_scope="word_embedding"):
            emb=embedding(input,
                           vocab_size=self.args.vocab_size,
                           num_units=self.args.emb_size,
                           scale=True,
                           scope="embed",reuse=reuse)

        return emb

    def build_encoder(self,input,reuse):
        word_emb = self.build_emb(input, reuse=reuse)
        word_emb = tf.nn.dropout(word_emb, self.dropout)
        with tf.variable_scope(self.scope + "_input_highway", reuse=reuse):
            input_dim = self.args.emb_size
            sent_repres = match_utils.multi_highway_layer(word_emb, input_dim, self.args.highway_layer_num)
        return sent_repres

    def build_interactor(self, sent1_repres, sent2_repres, sent1_len, sent2_len,
                         sent1_mask, sent2_mask, *args, **kargs):

        reuse = kargs["reuse"]
        input_dim = sent1_repres.get_shape()[-1]
        with tf.variable_scope(self.scope + "_interaction_module", reuse=reuse):
            (match_representation, match_dim) = match_utils.bilateral_match_func(sent1_repres, sent2_repres,
                                                                                 sent1_len,
                                                                                 sent2_len,
                                                                                 tf.cast(sent1_mask, tf.float32),
                                                                                 tf.cast(sent2_mask, tf.float32),
                                                                                 input_dim, True,
                                                                                 options=self.args)

            return match_representation, match_dim

    def build_predictor(self, matched_repres, *args, **kargs):
        match_dim = kargs["match_dim"]
        reuse = kargs["reuse"]
        num_classes = self.args.num_classes
        # dropout_rate = tf.cond(self.is_training,
        #                        lambda: self.config.dropout_rate,
        #                        lambda: 0.0)

        with tf.variable_scope(self.scope + "_prediction_module", reuse=reuse):
            # ========Prediction Layer=========
            # match_dim = 4 * self.options.aggregation_lstm_dim
            w_0 = tf.get_variable("w_0", [match_dim, match_dim / 2], dtype=tf.float32)
            b_0 = tf.get_variable("b_0", [match_dim / 2], dtype=tf.float32)
            w_1 = tf.get_variable("w_1", [match_dim / 2, num_classes], dtype=tf.float32)
            b_1 = tf.get_variable("b_1", [num_classes], dtype=tf.float32)

            # if is_training: match_representation = tf.nn.dropout(match_representation, (1 - options.dropout_rate))
            logits = tf.matmul(matched_repres, w_0) + b_0
            logits = tf.tanh(logits)
            logits = tf.nn.dropout(logits, (1 - self.dropout))

            self.estimation = tf.matmul(logits, w_1) + b_1
            self.pred_probs = tf.nn.softmax(self.estimation)

    def build_model(self, *args, **kargs):
        with tf.device('/device:GPU:%s' % gpu_id):
            self.sent1_encoded = self.build_encoder(self.sent1_word, reuse=False)
            self.sent2_encoded = self.build_encoder(self.sent2_word, reuse=True)

            [self.aggregat_repres,
             match_dim] = self.build_interactor(self.sent1_encoded,
                                                self.sent2_encoded,
                                                self.sent1_len,
                                                self.sent2_len,
                                                self.sent1_token_mask,
                                                self.sent2_token_mask,
                                                reuse=None)
            self.build_predictor(self.aggregat_repres,
                                 reuse=None,
                                 match_dim=match_dim)

