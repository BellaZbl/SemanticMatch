
from utils import *
from model.utils.cnn_utils import *
import numpy as np
import tensorflow as tf
from loss import point_wise_loss
from model.model_template import ModelTemplate
import os




class SiameseCNN(ModelTemplate):
    def __init__(self,args,scope):
        super(SiameseCNN, self).__init__(args,scope)
        self.args=args
        self.scope=scope
        self.args.sent1_psize=3
        self.args.sent2_psize=3
        self.args.num_layers=1
        self.args.l2_reg=0.001
        self.args.filter=4
        self.args.filter_num= 16
        self.args.pool_size = 5
        self.args.out_channel = 16
        self.args.model_type='BCNN'#BCNN, ABCNN1, ABCNN2, ABCNN3
        self.args.l2_reg=0.001

    def build_placeholder(self, ):
        with tf.variable_scope(name_or_scope="word_embedding"):
            with tf.variable_scope(name_or_scope=self.scope):
                self.sent1_word = tf.placeholder(shape=(None, self.args.seq_len), dtype=tf.int32, name='sent1_word')
                self.sent1_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='sent1_len')
                self.sent2_word = tf.placeholder(shape=(None, self.args.seq_len), dtype=tf.int32, name='sent2_word')
                self.sent2_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='sent2_len')
                self.dropout = tf.placeholder(dtype=tf.float32)
                self.target = tf.placeholder(shape=(None), dtype=tf.int32, name='target')
                self.sent1_token_mask = tf.cast(self.sent1_word, tf.bool)
                self.sent2_token_mask = tf.cast(self.sent2_word, tf.bool)
                self.dpool_index = tf.placeholder(tf.int32, name='dpool_index',
                                                  shape=(None, self.args.seq_len,self.args.seq_len , 3))

    def build_emb(self, input, reuse=False, *args, **kwargs):
        with tf.variable_scope(name_or_scope="word_embedding",reuse=reuse):
            emb = self.embedding(input,
                            vocab_size=self.args.vocab_size,
                            num_units=self.args.emb_size,
                            scale=True,
                            scope="embed", reuse=reuse)

        return emb

    def transformer_encoder(self,enc1,enc2,reuse=False):
        enc = multihead_attention(queries=enc1,
                                         keys=enc2,
                                         num_units=self.args.emb_size,
                                         num_heads=4,
                                         dropout_rate=0.9,
                                         is_training=True,
                                         causality=False, reuse=reuse)

        return enc

    def cnn_encoder(self,input_tensor,name,reuse,index):
        with tf.variable_scope(name_or_scope=name,reuse=reuse):
            if index==0:
                in_channel=1
            else:
                in_channel=self.args.out_channel
            weight=tf.Variable(tf.random_normal(shape=[self.args.filter,self.args.filter,in_channel,self.args.out_channel]))
            conv=tf.nn.conv2d(input=input_tensor, filter=weight, strides=[1,1,1,1], padding="SAME",)
            conv1=tf.nn.max_pool(conv, ksize=[1, self.args.pool_size, self.args.pool_size, 1], strides=[1, 2, 2, 1],padding='VALID')

            return conv1

    def build_model(self):
        with tf.device('/device:GPU:%s' % self.gpu_id):

            self.s1_emb=self.build_emb(self.sent1_word,reuse=False)
            self.s2_emb=self.build_emb(self.sent2_word,reuse=True)

            self.s1_emb=self.transformer_encoder(self.s1_emb,self.s2_emb,reuse=False)
            self.s2_emb=self.transformer_encoder(self.s2_emb,self.s1_emb,reuse=True)

            if self.args.cosine:
                x1_expanded = tf.expand_dims(self.s1_emb, -1)
                x2_expanded = tf.expand_dims(self.s2_emb, -1)
                for i in range(self.args.num_layers):
                    x1_expanded = self.cnn_encoder(input_tensor=x1_expanded, name='conv_%s' % str(i), reuse=False, index=i)
                    x2_expanded = self.cnn_encoder(input_tensor=x2_expanded, name='conv_%s' % str(i), reuse=True, index=i)
                s1_emb = tf.layers.flatten(x1_expanded)
                s2_emb = tf.layers.flatten(x2_expanded)
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
                transpose_s1_emb = tf.transpose(self.s1_emb,[0, 2, 1])  # batch_size x sequence_length x words_dimensionality
                transpose_s2_emb = tf.transpose(self.s2_emb, [0, 2, 1])
                x1_expanded = tf.expand_dims(transpose_s1_emb, -1)
                x2_expanded = tf.expand_dims(transpose_s2_emb, -1)

                LI_1, LO_1, RI_1, RO_1 = CNN_layer(variable_scope=self.scope + "CNN-0", x1=x1_expanded, x2=x2_expanded,
                                                   s=self.args.seq_len, d=self.args.emb_size, w=self.args.filter, di=self.args.filter_num,
                                                   model_type=self.args.model_type, l2_reg=self.args.l2_reg)

                if self.args.num_layers > 1:
                    LI_2, LO_2, RI_2, RO_2 = CNN_layer(variable_scope=self.scope + "CNN-1", x1=LI_1, x2=RI_1,
                                                       s=self.args.seq_len,
                                                       d=self.args.filter_num, w=self.args.filter, di=self.args.filter_num, model_type=self.args.model_type,
                                                       l2_reg=self.args.l2_reg)
                    h1 = LO_2
                    h2 = RO_2
                    print("------second cnn layer---------")
                else:
                    h1 = LO_1
                    h2 = RO_1


                with tf.variable_scope("feature_layer"):

                    print(h1.get_shape())

                    self.output_features = tf.concat([h1, h2, h1 - h2, h1 * h2], axis=1, name="output_features")
                    #
                    # self.enc_1 = multihead_attention(queries=self.s1_emb,
                    #                                keys=self.s2_emb,
                    #                                num_units=self.args.emb_size,
                    #                                num_heads=4,
                    #                                dropout_rate=0.9,
                    #                                is_training=True,
                    #                                causality=False,reuse=False)
                    #
                    # self.enc_2 = multihead_attention(queries=self.s2_emb,
                    #                                keys=self.s1_emb,
                    #                                num_units=self.args.emb_size,
                    #                                num_heads=4,
                    #                                dropout_rate=0.9,
                    #                                is_training=True,
                    #                                causality=False,reuse=True)
                    #
                    # enc1=last_relevant_output(self.enc_1,self.sent1_len)
                    # enc2=last_relevant_output(self.enc_2,self.sent2_len)
                    #
                    # enc=tf.concat([enc1,enc2],1)
                    # self.output_features=tf.concat([self.output_features,enc],1)

                with tf.variable_scope("output_layer"):

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


