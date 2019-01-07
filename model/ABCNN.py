
from utils import *
from model.utils.abcnn_utils import *
import numpy as np
import tensorflow as tf
from loss import point_wise_loss
from model.model_template import ModelTemplate
import os




class ABCNN(ModelTemplate):
    def __init__(self,args,scope):
        super(ABCNN, self).__init__(args,scope)
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
        # self.args.model_type='BCNN'#BCNN, ABCNN1, ABCNN2, ABCNN3
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



            transpose_s1_emb = tf.transpose(self.s1_emb,[0, 2, 1])  # batch_size x sequence_length x words_dimensionality
            transpose_s2_emb = tf.transpose(self.s2_emb, [0, 2, 1])
            x1_expanded = tf.expand_dims(transpose_s1_emb, -1)
            x2_expanded = tf.expand_dims(transpose_s2_emb, -1)
            #Lwp: left wide pooling
            #Lap: left all pooling
            Lwp, Lap, Rwp, Rap = CNN_layer(variable_scope=self.scope + "CNN-0", x1=x1_expanded, x2=x2_expanded,
                                               s=self.args.seq_len, d=self.args.emb_size, w=self.args.filter, di=self.args.filter_num,
                                               model_type=self.args.model_type, l2_reg=self.args.l2_reg)

            if self.args.num_layers > 1:
                Lwp, Lap, Rwp, Rap = CNN_layer(variable_scope=self.scope + "CNN-1", x1=Lwp, x2=Rwp,
                                                   s=self.args.seq_len,
                                                   d=self.args.filter_num, w=self.args.filter, di=self.args.filter_num, model_type=self.args.model_type,
                                                   l2_reg=self.args.l2_reg)
                h1 = Lap
                h2 = Rap
                print("------second cnn layer---------")
            else:
                h1 = Lap
                h2 = Rap


            with tf.variable_scope("feature_layer"):

                print(h1.get_shape())

                self.output_features = tf.concat([h1, h2, h1 - h2, h1 * h2], axis=1, name="output_features")

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


