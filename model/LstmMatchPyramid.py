import numpy as np
import tensorflow as tf
from utils import *
from model.utils.matchpyramid import rnn_utils,match_utils
from loss import point_wise_loss
import time
import tqdm
import os
gpu_id=0
os.environ["CUDA_VISIBLE_DEVICES"] = "%s"%gpu_id

class LstmMatchPyramid(object):
    def __init__(self,args,scope):
        self.args=args
        self.scope=scope

        self.args.sent1_psize=3
        self.args.sent2_psize=3
        self.args.num_layers=1
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



    def build_model(self):
        with tf.device('/device:GPU:%s' % gpu_id):

            self.s1_emb=self.build_emb(self.sent1_word,reuse=False)
            self.s2_emb=self.build_emb(self.sent2_word,reuse=True)
            self.s1_emb=sent_encoder(self.s1_emb,200,self.sent1_len,name='bilstm_encoder',dropout=0.9,reuse=False)
            self.s2_emb=sent_encoder(self.s2_emb,200,self.sent2_len,name='bilstm_encoder',dropout=0.9,reuse=True)

            # batch_size * X1_maxlen * X2_maxlen
            self.cross = tf.einsum('abd,acd->abc', self.s1_emb, self.s2_emb)
            self.cross_img = tf.expand_dims(self.cross, 3)

            with tf.variable_scope(self.scope + '_conv_pooling_layer'):
                # convolution
                self.w1=tf.Variable(tf.random_normal(shape=[2,10,1,8],mean=0.0,stddev=0.2,dtype=tf.float32),name='w1')
                # self.w1 = tf.get_variable('w1',
                #                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.2,
                #                                                                       dtype=tf.float32),
                #                           dtype=tf.float32,
                #                           shape=[2, 10, 1, 8])
                self.b1=tf.Variable(tf.constant(value=0.0,shape=[8]),name='b1')
                # self.b1 = tf.get_variable('b1', initializer=tf.constant_initializer(0.0), dtype=tf.float32, shape=[8])
                # batch_size * X1_maxlen * X2_maxlen * feat_out
                self.conv1 = tf.nn.relu(tf.nn.conv2d(self.cross_img, self.w1, [1, 1, 1, 1], "SAME") + self.b1)

                # dynamic pooling
                self.conv1_expand = tf.gather_nd(self.conv1, self.dpool_index) # [batch_size,seq_len,seq_len,channel] gather_nd:允许在多维上进行索引
                print(self.conv1_expand)
                self.pool1 = tf.nn.max_pool(self.conv1_expand,
                                            [1, self.args.seq_len / self.args.sent1_psize,
                                             self.args.seq_len / self.args.sent2_psize, 1],
                                            [1, self.args.seq_len / self.args.sent1_psize,
                                             self.args.seq_len / self.args.sent2_psize, 1], "VALID") # [batch_size,3,3,8]

                print(self.pool1)

            with tf.variable_scope(self.scope + '_feature_layer'):
                flatten = tf.reshape(self.pool1,
                                     [-1, self.args.sent1_psize * self.args.sent2_psize * 8])
                print(flatten.get_shape())

                self.output_features = rnn_utils.highway_net(flatten,
                                                             self.args.num_layers,
                                                             0.9,
                                                             batch_norm=False,
                                                             training=True)

            with tf.variable_scope(self.scope + "_output_layer"):
                self.estimation = tf.contrib.layers.fully_connected(
                    inputs=self.output_features,
                    num_outputs=self.args.num_classes,
                    activation_fn=None,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    weights_regularizer=tf.contrib.layers.l2_regularizer(scale=self.args.l2_reg),
                    biases_initializer=tf.constant_initializer(1e-04),
                    scope="FC")

            match_utils.add_reg_without_bias(self.scope)
            self.pred_probs = tf.contrib.layers.softmax(self.estimation)
            # self.logits = tf.cast(tf.argmax(self.pred_probs, -1), tf.int32)

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


    def infer(self,infer_data_loader,restore_model_path):
        start_time = time.time()
        self.saver = tf.train.Saver()
        id2word=self.args.id2word
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        # _logger.info('entity_words:%s'%(entity_dict.keys()))
        if restore_model_path:
            self.saver.restore(self.sess, restore_model_path)
            print("restore model from ", restore_model_path)
        else:
            print('error!!please input restore_model_path')

        for i, data in enumerate(infer_data_loader):
            feed_dict = {
                self.sent1_word: data["sent1_word"],
                self.sent2_word: data["sent2_word"],
                self.sent1_len: data["sent1_len"],
                self.sent2_len: data["sent2_len"],
            }
            infer_pred_label,infer_pred_probs=self.sess.run([self.pred_label,self.pred_probs],feed_dict=feed_dict)

            # cnadidate_list
            sent2_word=data["sent2_word"]
            sent1_word=data["sent1_word"]
            for i in range(len(data["sent1_word"])):
                sent1 = ' '.join([id2word[e] for e in np.array(sent1_word[i]) if e != 0])
                sent2 = ' '.join([id2word[e] for e in np.array(sent2_word[i]) if e != 0])
                prob = infer_pred_probs[i]
                cand_label = infer_pred_label[i]
                print(sent1, '###', sent2, "###", prob, '###', cand_label, '\n')