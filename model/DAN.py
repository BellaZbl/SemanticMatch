import numpy as np
import tensorflow as tf
from utils import *
from model.utils.cnn_utils import *
from model.utils.rnn_utils import *
from model.utils.matchpyramid import rnn_utils,match_utils
from loss import point_wise_loss
import time
import tqdm
import os
gpu_id=0
os.environ["CUDA_VISIBLE_DEVICES"] = "%s"%gpu_id

class SiameseCNN(object):
    def __init__(self,args,scope):
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
        with tf.variable_scope(name_or_scope="word_embedding"):
            emb = embedding(input,
                            vocab_size=self.args.vocab_size,
                            num_units=self.args.emb_size,
                            scale=True,
                            scope="embed", reuse=reuse)

        return emb

    def _build_network(self, emb, emb_mask, emb_len):
        with tf.variable_scope("self-attention-dan") as scope:
            emb = tf.nn.dropout(emb, self.dropout)
            encoder_fw, encoder_bw = rnn_utils.BiRNN(emb, self.dropout,
                                                     "tied_encoder",
                                                     emb_len,
                                                     self.args.hidden,
                                                     self.is_train, self.rnn_cell)

            encoder_represnetation = tf.concat([encoder_fw, encoder_bw], axis=-1)
            emd_mask = tf.cast(emb_mask, tf.float32)
            representation = rnn_utils.task_specific_attention(encoder_represnetation,
                                                               self.hidden_units,
                                                               emb_mask)

            deep_self_att = rnn_utils.highway_net(representation,
                                                  self.config["num_layers"],
                                                  self.dropout_keep_prob,
                                                  batch_norm=False,
                                                  training=self.is_train)
            return deep_self_att

    def build_model(self):
        with tf.device('/device:GPU:%s' % gpu_id):

            self.s1_emb=self.build_emb(self.sent1_word,reuse=False)
            self.s2_emb=self.build_emb(self.sent2_word,reuse=True)

            transpose_s1_emb = tf.transpose(self.s1_emb,[0, 2, 1])  # batch_size x sequence_length x words_dimensionality
            transpose_s2_emb = tf.transpose(self.s2_emb, [0, 2, 1])
            x1_expanded = tf.expand_dims(transpose_s1_emb, -1)
            x2_expanded = tf.expand_dims(transpose_s2_emb, -1)

            LI_1, LO_1, RI_1, RO_1 = CNN_layer(variable_scope=self.scope + "CNN-0", x1=x1_expanded, x2=x2_expanded,
                                               s=self.args.seq_len, d=self.args.emb_size, w=self.args.filter, di=self.args.filter_num,
                                               model_type=self.args.model_type, l2_reg=self.args.l2_reg)

            if self.args.num_layers > 1:
                LI_2, LO_2, RI_2, RO_2 =  CNN_layer(variable_scope=self.scope + "CNN-0", x1=x1_expanded, x2=x2_expanded,
                                               s=self.args.seq_len, d=self.args.emb_size, w=self.args.filter, di=self.args.filter_num,
                                               model_type=self.args.model_type, l2_reg=self.args.l2_reg)
                h1 = LO_2
                h2 = RO_2
                print("------second cnn layer---------")
            else:
                h1 = LO_1
                h2 = RO_1

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


    def build_accuracy(self, *args, **kargs):
        self.pred_label = tf.argmax(self.pred_probs, axis=-1)
        correct = tf.equal(
            tf.cast(self.pred_label, tf.int32),
            tf.cast(self.target, tf.int32)
        )
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    def build_loss(self, *args, **kargs):
        with tf.device('/device:GPU:%s' % gpu_id):
            if self.args.loss == "softmax_loss":
                self.loss, _ = point_wise_loss.softmax_loss(self.estimation, self.target,
                                                            *args, **kargs)
            elif self.args.loss == "sparse_amsoftmax_loss":
                self.loss, _ = point_wise_loss.sparse_amsoftmax_loss(self.estimation, self.target,
                                                                     self.args, *args, **kargs)
            elif self.args.loss == "focal_loss_binary_v2":
                self.loss, _ = point_wise_loss.focal_loss_binary_v2(self.estimation, self.target,
                                                                    self.args, *args, **kargs)
    def build_op(self):
        with tf.device('/device:GPU:%s' % gpu_id):
            if self.args.opt == 'adadelta':
                self.optimizer = tf.train.AdadeltaOptimizer(self.args.lr).minimize(self.loss)
            elif self.args.opt == 'adam':
                self.optimizer = tf.train.AdamOptimizer(self.args.lr).minimize(self.loss)
            elif self.args.opt == 'rmsprop':
                self.optimizer = tf.train.RMSPropOptimizer(self.args.lr).minimize(self.loss)
            # self.optimizer.minimize(self.loss)


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

    def get_feed_dict(self, sample_batch, dropout_keep_prob=None, data_type='train'):

        feed_dict = {
            self.sent1_word: sample_batch["sent1_word"],
            self.sent2_word: sample_batch["sent2_word"],
            self.sent1_len: sample_batch["sent1_len"],
            self.sent2_len: sample_batch["sent2_len"],
            self.target: sample_batch["label"],
            self.dpool_index: self.dynamic_pooling_index(np.array(sample_batch["sent1_len"]), np.array(sample_batch["sent2_len"]),
                                                         self.args.seq_len, self.args.seq_len),
        }

        return feed_dict

    def iteration(self, sess,epoch, data_loader, train=True):
        """

        :param epoch:
        :param data_loader:
        :param train:
        :return:
        """
        if train:flag='train'
        else:flag='test'

        pbar = tqdm.tqdm(total=len(data_loader))

        avg_loss = 0.0
        total_correct = 0
        total_element = 0
        avg_acc=0.0
        for i, data in enumerate(data_loader):
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value for key, value in data.items()}
            feed_dict = self.get_feed_dict(data)
            feed_dict.update({self.dropout: 0.1})
            if train:
                _, loss,acc = sess.run([self.optimizer, self.loss,self.accuracy], feed_dict=feed_dict)
            else:
                loss,acc = sess.run([self.loss,self.accuracy], feed_dict=feed_dict)
            avg_loss += loss
            total_correct += 0
            total_element += 1
            avg_acc+=acc
            pbar.update(i)

            post_fix = {
                "mode":flag,
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                # "avg_acc": total_correct / total_element * 100,
                "loss": loss
            }
            # print(post_fix)
        print("EP%d_%s, avg_loss=  avg_acc="  % (epoch,flag), avg_loss / len(data_loader),avg_acc / len(data_loader))
        pbar.close()



    def train(self,train_data_loader,test_data_loader=None,restore_model=None,save_model=None):
        start_time=time.time()
        self.saver=tf.train.Saver()

        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess=tf.Session(config=config)

        # _logger.info('entity_words:%s'%(entity_dict.keys()))
        if restore_model:
            self.saver.restore(self.sess,restore_model)
            print("restore model from ",restore_model)
        else:
            self.sess.run(tf.global_variables_initializer())
        for epoch in range(self.args.epochs):
            self.iteration(self.sess,epoch,train_data_loader,True)
            self.iteration(self.sess,epoch,test_data_loader,False)
            if save_model:
                self.save(save_model)
        end_time=time.time()

        print('#'*20,end_time-start_time)

    def save(self,save_model):
        self.saver.save(self.sess,save_model)
        print('save in ',save_model)
