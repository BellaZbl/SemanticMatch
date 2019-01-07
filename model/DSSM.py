
from utils import *
from model.utils.cnn_utils import *
import numpy as np
import tensorflow as tf
from loss import point_wise_loss
from model.model_template import ModelTemplate
import os




class Dssm(ModelTemplate):
    def __init__(self,args,scope):
        super(Dssm, self).__init__(args,scope)
        self.args=args
        self.scope=scope
        self.args.sent1_psize=3
        self.args.sent2_psize=3
        self.args.num_layers=1
        self.args.l2_reg=0.001
        self.args.filter=4
        self.args.filter_num= 16
        self.args.model_type='BCNN'#BCNN, ABCNN1, ABCNN2, ABCNN3
        self.args.l2_reg=0.001
        self.args.dssm_dim=[300,600,300]

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

    def dssm_encoder(self,input,out_dim,reuse,name):

        with tf.variable_scope(name_or_scope=name,reuse=reuse):
            output=tf.layers.dense(input,out_dim,use_bias=True,reuse=reuse)
            return output

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

            s1_emb=self.build_emb(self.sent1_word,reuse=False)
            s2_emb=self.build_emb(self.sent2_word,reuse=True)
            for i,dim in enumerate(self.args.dssm_dim):
                s1_emb=self.dssm_encoder(s1_emb,dim,name="layer_%s"%i,reuse=False)
                s2_emb=self.dssm_encoder(s2_emb,dim,name="layer_%s"%i,reuse=True)

            # Cosine similarity
            s1_emb_=mean_pool(s1_emb,self.sent1_len)
            s2_emb_=mean_pool(s2_emb,self.sent2_len)

            query_norm = tf.sqrt(tf.reduce_sum(tf.square(s1_emb_), 1, True))
            doc_norm = tf.sqrt(tf.reduce_sum(tf.square(s2_emb_), 1, True))
            prod = tf.reduce_sum(tf.multiply(s1_emb_, s2_emb_), 1, True)
            norm_prod = tf.multiply(query_norm, doc_norm) + 0.01
            cos_sim = tf.truediv(prod, norm_prod)
            neg_cos_sim=tf.abs(1 - cos_sim)
            self.estimation=tf.concat([neg_cos_sim,cos_sim],1)
            print(self.estimation)
            # prod = tf.reduce_sum(tf.multiply(s1_emb, s2_emb), 1, True)
            # unprod = tf.abs(1 - prod)
            # self.out = tf.concat([unprod, prod], axis=1)

            # with tf.variable_scope("output_layer"):

                # self.estimation = tf.contrib.layers.fully_connected(
                #     inputs=self.output_features,
                #     num_outputs=self.args.num_classes,
                #     activation_fn=None,
                #     weights_initializer=tf.contrib.layers.xavier_initializer(),
                #     weights_regularizer=tf.contrib.layers.l2_regularizer(scale=self.args.l2_reg),
                #     biases_initializer=tf.constant_initializer(1e-04),
                #     scope="FC"
                # )

            self.pred_probs = tf.contrib.layers.softmax(self.estimation)
            self.logits = tf.cast(tf.argmax(self.pred_probs, -1), tf.int32)




