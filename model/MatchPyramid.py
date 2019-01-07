import numpy as np
import tensorflow as tf
from utils import *
from model.utils.matchpyramid.rnn_utils import *
from model.model_template import ModelTemplate
import os
gpu_id=0
os.environ["CUDA_VISIBLE_DEVICES"] = "%s"%gpu_id

class MatchPyramid(ModelTemplate):
    def __init__(self,args,scope):
        super(MatchPyramid, self).__init__(args,scope)

        self.args=args
        self.scope=scope

        self.args.sent1_psize=5
        self.args.sent2_psize=5
        self.args.num_layers=1
        self.args.l2_reg=0.001
        self.args.mp_num_filters=[8,64]
        self.args.mp_filter_sizes=[5,5]
        self.args.mp_pool_sizes=[5,5]

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

    def _interaction_semantic_feature_layer(self,emb_seq_left , emb_seq_right, seq_len_left, seq_len_right, granularity="word"):

        #dropout
        random_seed = np.random.randint(10000000)
        emb_seq_left = tf.layers.dropout(emb_seq_left,rate=self.dropout,seed=random_seed)
        random_seed = np.random.randint(10000000)
        emb_seq_right = tf.layers.dropout(emb_seq_right,rate=self.dropout,seed=random_seed)

        #encode

        # enc_seq_left = sent_encoder(emb_seq_left,self.args.hidden,seq_len_left,name='encod',reuse=False)
        # enc_seq_right = sent_encoder(emb_seq_right,self.args.hidden,seq_len_right,name='encod',reuse=True)
        enc_seq_left,enc_seq_right=emb_seq_left,emb_seq_right
        #attend
        # [batchsize, s1, s2]
        att_mat = tf.einsum("abd,acd->abc", enc_seq_left, enc_seq_right)

        att_seq_left = attend(enc_seq_left, context=att_mat, feature_dim=self.args.hidden,
                              method='ave',
                              scope_name='att_seq',
                              reuse=False)
        att_seq_right = attend(enc_seq_right, context=tf.transpose(att_mat), feature_dim=self.args.hidden,
                               method='ave',
                               scope_name='att_seq',
                               reuse=True)

        #### MLP nonlinear projection
        sem_seq_left = mlp_layer(att_seq_left, fc_type="fc",
                                 hidden_units=[self.args.hidden],
                                 dropouts=[self.dropout],
                                 scope_name='fc_mlp',
                                 reuse=False,
                                 training=True,
                                 seed=123)
        sem_seq_right = mlp_layer(att_seq_right, fc_type='fc',
                                  hidden_units=[self.args.hidden],
                                  dropouts=[self.dropout],
                                  scope_name='fc_mlp',
                                  reuse=True,
                                  training=True,
                                  seed=234)

        return emb_seq_left, enc_seq_left, att_seq_left, sem_seq_left, \
               emb_seq_right, enc_seq_right, att_seq_right, sem_seq_right



    def _semantic_feature_layer(self, emb_seq, seq_len, reuse=False):

        #### embed


        #### dropout
        random_seed = np.random.randint(10000000)
        emb_seq = tf.layers.dropout(emb_seq,rate=self.dropout)

        #### encode
        # enc_seq=sent_encoder(emb_seq,self.args.hidden,seq_len,name='encod',reuse=reuse)
        enc_seq=textcnn(emb_seq,num_layers=2,num_filters=8,filter_sizes=[3,6],reuse=reuse)

        return enc_seq

    def _mp_cnn_layer(self, cross, dpool_index, filters, kernel_size, pool_size, strides, name):
        cross_conv = tf.layers.conv2d(
            inputs=cross,
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation=tf.nn.relu,
            strides=1,
            reuse=False,
            name=name + "cross_conv")

        cross_conv = tf.gather_nd(cross_conv, dpool_index)
        cross_pool = tf.layers.max_pooling2d(
            inputs=cross_conv,
            pool_size=pool_size,
            strides=strides,
            padding="valid",
            name=name + "cross_pool")
        return cross_pool

    def _mp_semantic_feature_layer(self, match_matrix, dpool_index, granularity="word"):

        # conv-pool layer 1
        filters = self.args.mp_num_filters[0]
        kernel_size = self.args.mp_filter_sizes[0]
        seq_len = self.args.seq_len
        pool_size0 = self.args.mp_pool_sizes[0]
        pool_sizes = [seq_len / pool_size0, seq_len / pool_size0]
        strides = [seq_len / pool_size0, seq_len / pool_size0]
        conv1 = self._mp_cnn_layer(match_matrix, dpool_index, filters, kernel_size, pool_sizes, strides,
                                   name='conv1')
        conv1_flatten = tf.reshape(conv1, [-1, self.args.mp_num_filters[0] * (pool_size0 * pool_size0)])

        # conv-pool layer 2
        filters = self.args.mp_num_filters[1]
        kernel_size = self.args.mp_filter_sizes[1]
        pool_size1 = self.args.mp_pool_sizes[1]
        pool_sizes = [seq_len / pool_size1, seq_len / pool_size1]
        strides = [seq_len / pool_size1, seq_len / pool_size1]

        conv2 = self._mp_cnn_layer(match_matrix, dpool_index, filters, kernel_size, pool_sizes, strides,
                                   name="conv2")
        print(conv2,pool_size1)

        conv2_flatten = tf.reshape(conv2, [-1, self.args.mp_num_filters[1] * (pool_size1 * pool_size1)])

        # cross = tf.concat([conv1_flatten, conv2_flatten], axis=-1)

        return conv2_flatten


    def build_model(self):
        with tf.device('/device:GPU:%s' % gpu_id):

            self.s1_emb=self.build_emb(self.sent1_word,reuse=False)
            self.s2_emb=self.build_emb(self.sent2_word,reuse=True)

            # self.s1_emb=self._semantic_feature_layer(self.s1_emb,self.sent1_len,reuse=False)
            # self.s2_emb=self._semantic_feature_layer(self.s2_emb,self.sent2_len,reuse=True)

            # emb_seq_left, enc_seq_left, att_seq_left, sem_seq_left,emb_seq_right, enc_seq_right, att_seq_right, sem_seq_right\
            #     =self._interaction_semantic_feature_layer(self.s1_emb,self.s2_emb,self.sent1_len,self.sent2_len)
            # batch_size * X1_maxlen * X2_maxlen dot function
            self.cross = tf.einsum('abd,acd->abc', self.s1_emb, self.s2_emb)
            self.cross_img = tf.expand_dims(self.cross, 3)

            with tf.variable_scope(self.scope + '_conv_pooling_layer'):
                # convolution
                ### new function
                # flatten=self._mp_semantic_feature_layer(self.cross_img,self.dpool_index)

                ### old function
                self.w1=tf.Variable(tf.random_normal(shape=[5,5,1,64],mean=0.0,stddev=0.2,dtype=tf.float32),name='w1')
                self.b1=tf.Variable(tf.constant(value=0.0,shape=[64]),name='b1')
                # batch_size * X1_maxlen * X2_maxlen * feat_out
                # self.conv1 = tf.nn.relu(tf.nn.conv2d(self.cross_img, self.w1, [1, 1, 1, 1], "SAME") + self.b1)
                self.conv1 = tf.nn.conv2d(self.cross_img, self.w1, [1, 1, 1, 1], "SAME") + self.b1
                # dynamic pooling
                self.conv1_expand = tf.gather_nd(self.conv1, self.dpool_index) # [batch_size,seq_len,seq_len,channel] gather_nd:允许在多维上进行索引
                self.pool1 = tf.nn.max_pool(self.conv1_expand,
                                            [1, self.args.seq_len / self.args.sent1_psize,
                                             self.args.seq_len / self.args.sent2_psize, 1],
                                            [1, self.args.seq_len / self.args.sent1_psize,
                                             self.args.seq_len / self.args.sent2_psize, 1], "VALID") # [batch_size,3,3,8]

                with tf.variable_scope('fc1'):
                    print(self.pool1)
                    flatten= tf.reshape(self.pool1, [-1, self.args.sent1_psize * self.args.sent2_psize * 64])
                    # flatten = tf.nn.relu(tf.contrib.layers.linear(
                    #     tf.reshape(self.pool1, [-1, self.args.sent1_psize * self.args.sent2_psize * 64]),
                    #     1600))
                    # pool=tf.nn.relu(self.pool1)
                    # pool=self.pool1
                    # flatten = tf.contrib.layers.linear(
                    #     tf.reshape(pool, [-1, self.args.sent1_psize * self.args.sent2_psize * 64]),
                    #     1600)
                    # self.output_features=flatten
                    self.output_features=flatten
                    # self.output_features = highway_net(flatten,2,self.dropout,
                    #                                              batch_norm=False,
                    #                                              training=False)

            # with tf.variable_scope(self.scope + '_feature_layer'):
            #     # flatten = tf.reshape(self.pool1,
            #     #                      [-1, self.args.sent1_psize * self.args.sent2_psize * 8])
            #
            #     self.output_features=tf.layers.dense(flatten,self.args.num_classes)
            #     # self.output_features = rnn_utils.highway_net(flatten,
            #     #                                              self.args.num_layers,
            #     #                                              0.9,
            #     #                                              batch_norm=False,
            #     #                                              training=True)
            #
            #     self.estimation=self.output_features

            with tf.variable_scope(self.scope + "_output_layer"):
                self.estimation = tf.contrib.layers.fully_connected(
                    inputs=self.output_features,
                    num_outputs=self.args.num_classes,
                    activation_fn=None,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    weights_regularizer=tf.contrib.layers.l2_regularizer(scale=self.args.l2_reg),
                    biases_initializer=tf.constant_initializer(1e-04),
                    scope="FC")
            # print(self.estimation)
            # print(self.args.num_classes)
            # match_utils.add_reg_without_bias(self.scope)
            self.pred_probs = tf.contrib.layers.softmax(self.estimation)
            # self.logits = tf.cast(tf.argmax(self.pred_probs, -1), tf.int32)




