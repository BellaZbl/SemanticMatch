
from utils import *
from model.utils.cnn_utils import *
import numpy as np
import tensorflow as tf
from loss import point_wise_loss
from model.model_template import ModelTemplate
import os




class ARC_II(ModelTemplate):
    def __init__(self,args,scope):
        super(ARC_II, self).__init__(args,scope)
        self.args=args
        self.scope=scope
        self.args.sent1_psize=3
        self.args.sent2_psize=3
        self.args.num_layers=1
        self.args.l2_reg=0.001
        self.args.filter=15
        self.args.pool_size=5
        self.args.out_channel=16
        self.args.filter_num= 50
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
            emb = embedding(input,
                            vocab_size=self.args.vocab_size,
                            num_units=self.args.emb_size,
                            scale=True,
                            scope="embed", reuse=reuse)

        return emb

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
            s1_emb=tf.expand_dims(s1_emb,-1)
            s2_emb=tf.expand_dims(s2_emb,-1)

            for i in range(self.args.num_layers):
                s1_emb=self.cnn_encoder(input_tensor=s1_emb,name='conv_%s'%str(i),reuse=False,index=i)
                s2_emb=self.cnn_encoder(input_tensor=s2_emb,name='conv_%s'%str(i),reuse=True,index=i)
            pool1=tf.concat((s1_emb,s2_emb),1)

            self.output_features=tf.reshape(pool1,[-1,pool1.get_shape()[1]*pool1.get_shape()[2]*pool1.get_shape()[3]])

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


