import numpy as np
import tensorflow as tf
from utils import *
from model.utils.cnn_utils import *
from model.utils.rnn_utils import *
from collections import defaultdict
from model.model_template import ModelTemplate
from model.utils.matchpyramid import rnn_utils,match_utils
from loss import point_wise_loss
import time
import tqdm
import os
import torch


class TransformerModel(ModelTemplate):
    def __init__(self,args,scope):
        super(TransformerModel, self).__init__(args,scope)
        self.args=args
        self.scope=scope
        self.args.sent1_psize=3
        self.args.sent2_psize=3
        self.args.num_layers=1
        self.args.l2_reg=0.001
        self.args.filter=4
        self.args.filter_num= 50
        self.args.model_type='BCNN'#BCNN, ABCNN1, ABCNN2, ABCNN3
        self.args.l2_reg=0.001
        self.args.sinusoid=False
        self.args.num_blocks=6
        self.args.num_heads=4

    def build_placeholder(self, ):
        with tf.variable_scope(name_or_scope="word_embedding"):
            with tf.variable_scope(name_or_scope=self.scope):
                self.sent1_word = tf.placeholder(shape=(None, self.args.seq_len), dtype=tf.int32, name='sent1_word')
                self.sent1_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='sent1_len')
                self.sent2_word = tf.placeholder(shape=(None, self.args.seq_len), dtype=tf.int32, name='sent2_word')
                self.sent2_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='sent2_len')
                self.dropout = tf.placeholder(dtype=tf.float32)
                self.target = tf.placeholder(shape=(None,), dtype=tf.int32, name='target')
                self.sent1_token_mask = tf.cast(self.sent1_word, tf.bool)
                self.sent2_token_mask = tf.cast(self.sent2_word, tf.bool)


    def build_emb(self, sent_word, reuse=False, *args, **kwargs):
        with tf.variable_scope(name_or_scope="word_embedding",reuse=reuse):
            encoder_input_emb = self.embedding(sent_word,
                            vocab_size=self.args.vocab_size,
                            num_units=self.args.emb_size,
                            scale=True,
                            scope="embed", reuse=reuse)

            if self.args.sinusoid:
                encoder_input_emb += positional_encoding(sent_word,
                                                              num_units=self.args.emb_size,
                                                              zero_pad=False,
                                                              scale=False,
                                                              scope="enc_pe",reuse=reuse)
            else:
                encoder_input_emb += embedding(
                    tf.tile(tf.expand_dims(tf.range(tf.shape(sent_word)[1]), 0),
                            [tf.shape(sent_word)[0], 1]),
                    vocab_size=self.args.seq_len,
                    num_units=self.args.emb_size,
                    zero_pad=False,
                    scale=False,
                    scope="enc_pe",reuse=reuse)

            ## Dropout
            encoder_input_emb = tf.layers.dropout(encoder_input_emb,
                                                       rate=self.dropout,
                                                       training=True)

            return encoder_input_emb

    def transformer_encoder(self,encoder_emb1,encoder_emb2,name,reuse=False,num_blocks=4):
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            data=[]
            for i in range(self.args.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    enc = multihead_attention(queries=encoder_emb1,
                                                   keys=encoder_emb2,
                                                   num_units=self.args.emb_size,
                                                   num_heads=self.args.num_heads,
                                                   dropout_rate=self.dropout,
                                                   is_training=True,
                                                   causality=False)

                    ### Feed Forward
                    enc = feedforward(enc, num_units=[4 * self.args.emb_size, self.args.emb_size])
                    data.append(enc)
            return data[-1]

    def cosine_array(self,s1, s2):
        sim = tf.matmul(s1, tf.transpose(s2))
        s1_l2 = tf.linalg.norm(s1, 2, 1)
        s2_l2 = tf.linalg.norm(s2, 2, 1)
        s1_l2 = tf.expand_dims(s1_l2, -1)
        s2_l2 = tf.expand_dims(s2_l2, -1)
        fm = tf.matmul(s1_l2, tf.transpose(s2_l2))
        cosin_sim = sim / fm
        return cosin_sim

    def build_model(self):
        with tf.device('/device:GPU:%s' % self.gpu_id):
            s1_emb=self.build_emb(self.sent1_word,reuse=False)
            s2_emb=self.build_emb(self.sent2_word,reuse=True)
            s1_emb=self.transformer_encoder(s1_emb,s1_emb,name='self_trans',reuse=False,num_blocks=4)
            s2_emb=self.transformer_encoder(s2_emb,s2_emb,name='self_trans',reuse=True,num_blocks=4)

            s12_emb=self.transformer_encoder(s1_emb,s2_emb,name='inter_trans',reuse=False,num_blocks=4)
            s21_emb=self.transformer_encoder(s2_emb,s1_emb,name='inter_trans',reuse=True,num_blocks=4)

            s1_emb=mean_pool(s12_emb,self.sent1_len)
            s2_emb=mean_pool(s21_emb,self.sent2_len)

            if self.args.cosine:
               # print('s1_emb',s1_emb)
               # cosine=cosine_distance(s1_emb,s2_emb)
               # print('cosine',cosine)
               # cosine_0=tf.where(cosine<0.5, cosine, 1 - cosine)
               # cosine_1=tf.where(cosine>=0.5, cosine, 1 - cosine)
               # cosine_0=tf.expand_dims(cosine_0,1)
               # cosine_1=tf.expand_dims(cosine_1,1)
               # self.estimation=tf.concat([cosine_0,cosine_1],1)
               # print(self.estimation)
               query_norm = tf.sqrt(tf.reduce_sum(tf.square(s1_emb), 1, True))
               doc_norm = tf.sqrt(tf.reduce_sum(tf.square(s2_emb), 1, True))
               prod = tf.reduce_sum(tf.multiply(s1_emb, s2_emb), 1, True)
               norm_prod = tf.multiply(query_norm, doc_norm) + 0.01
               cos_sim = tf.truediv(prod, norm_prod)
               neg_cos_sim = tf.abs(1 - cos_sim)
               self.estimation = tf.concat([neg_cos_sim, cos_sim], 1)
               self.pred_probs = tf.contrib.layers.softmax(self.estimation)
               self.logits = tf.cast(tf.argmax(self.pred_probs, -1), tf.int32)

            else:
                self.output_features = tf.concat([s1_emb, s2_emb, s1_emb - s2_emb, s1_emb * s2_emb], axis=1, name="output_features")

                self.estimation = tf.contrib.layers.fully_connected(
                    inputs=self.output_features,
                    num_outputs=self.args.num_classes,
                    activation_fn=None,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    weights_regularizer=tf.contrib.layers.l2_regularizer(scale=self.args.l2_reg),
                    biases_initializer=tf.constant_initializer(1e-04),
                    scope="FC"
                )

                self.pred_probs = tf.contrib.layers.softmax(self.estimation)
                self.logits = tf.cast(tf.argmax(self.pred_probs, -1), tf.int32)

            # # y_true = tf.cast(self.target, tf.float32)
            # # self.distance_loss = y_true * (pred_probs) + (1 - y_true) * (tf.maximum((0.05 - pred_probs), 0))
            # pred_probs=tf.expand_dims(pred_probs,-1)
            # self.pred_probs=tf.concat([1.0-pred_probs,pred_probs],1)
            # # self.estimation=self.pred_probs
            # print(self.pred_probs)


    def dynamic_pooling_index(self, len1, len2, max_len1, max_len2):
        def dpool_index_(batch_idx, len1_one, len2_one, max_len1, max_len2):
            stride1 = 1.0 * max_len1 / len1_one
            stride2 = 1.0 * max_len2 / len2_one
            idx1_one = [int(i / stride1) for i in range(max_len1)]
            idx2_one = [int(i / stride2) for i in range(max_len2)]
            mesh1, mesh2 = np.meshgrid(idx1_one, idx2_one)
            index_one = np.transpose(np.stack([np.ones(mesh1.shape) * batch_idx, mesh1, mesh2]), (2, 1, 0))
            return index_one

        index = []
        for i in range(len(len1)):
            index.append(dpool_index_(i, len1[i], len2[i], max_len1, max_len2))

        return np.array(index)

