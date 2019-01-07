import numpy as np
import tensorflow as tf
from utils import *
from model.utils.cnn_utils import *
from model.utils.matchpyramid import rnn_utils,match_utils
from loss import point_wise_loss
import time
import tqdm
import os
import pickle as pkl
gpu_id=0
os.environ["CUDA_VISIBLE_DEVICES"] = "%s"%gpu_id

class SiameseCNNTriple(object):
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
        self.args.margin=0.5

    def build_placeholder(self, ):
        with tf.variable_scope(name_or_scope="word_embedding"):
            with tf.variable_scope(name_or_scope=self.scope):
                self.sent_0_word = tf.placeholder(shape=(None, self.args.seq_len), dtype=tf.int32, name='sent0_word')
                self.sent_0_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='sent0_len')

                self.sent_pos_word = tf.placeholder(shape=(None, self.args.seq_len), dtype=tf.int32, name='sent_pos_word')
                self.sent_pos_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='sent_pos_len')

                self.sent_neg_word = tf.placeholder(shape=(None, self.args.seq_len), dtype=tf.int32, name='sent_neg_word')
                self.sent_neg_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='sent_neg_len')

                self.dropout = tf.placeholder(dtype=tf.float32)
                self.sent_0_token_mask = tf.cast(self.sent_0_word, tf.bool)
                self.sent_pos_token_mask = tf.cast(self.sent_pos_word, tf.bool)
                self.sent_neg_token_mask = tf.cast(self.sent_neg_word, tf.bool)

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

    def cnn_encoder(self,x1_expanded,x2_expanded,reuse_1=False,reuse_2=False):
        LI_1, LO_1, RI_1, RO_1 = CNN_layer(variable_scope=self.scope + "CNN-0", x1=x1_expanded, x2=x2_expanded,
                                                   s=self.args.seq_len, d=self.args.emb_size, w=self.args.filter,
                                                   di=self.args.filter_num,
                                                   model_type=self.args.model_type, l2_reg=self.args.l2_reg,reuse=reuse_1)


        if self.args.num_layers > 1:
            LI_2, LO_2, RI_2, RO_2 = CNN_layer(variable_scope=self.scope + "CNN-1", x1=LI_1, x2=RI_1,
                                               s=self.args.seq_len,
                                               d=self.args.filter_num, w=self.args.filter, di=self.args.filter_num,
                                               model_type=self.args.model_type,
                                               l2_reg=self.args.l2_reg,reuse=reuse_2)
            h1 = LO_2
            h2 = RO_2
            print("------second cnn layer---------")
        else:
            h1 = LO_1
            h2 = RO_1
        return h1,h2

    def cnn_encoder_(self,input,name,reuse):
        with tf.variable_scope(name_or_scope=name,reuse=reuse):
            convd_w = tf.Variable(tf.random_uniform(shape=(4, self.args.emb_size, 1, 30), minval=-0.1, maxval=0.1),
                                  dtype=tf.float32)
            convd_b = tf.Variable(tf.random_uniform(shape=(30,), dtype=tf.float32))
            # strides 一个长为4的list. 表示每次卷积以后在input中滑动的距离 strides.shape=inputs.shape [batch_size,height,width,channels]
            # padding 有SAME和VALID两种选项，表示是否要保留不完全卷积的部分。如果是SAME，则保留
            # 转换shape cnn层的标准输入：[batch_size,height,width,channels]
            convd = tf.nn.conv2d(input, convd_w, strides=[1, 1, 1, 1], padding="SAME")  # 若滑动stride为1 代表输出维度和输入一致
            convd_1 = tf.nn.relu(tf.add(convd, convd_b))  # [batch_size,self.title_len,self.init_dim,out_channels]
            convd_pool_1 = tf.nn.max_pool(convd_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                          padding="SAME")  # size=[batch_size,title_len/2,init_din/2,20]
            return convd_pool_1

    def full_connect(self,input,out_size,name,reuse):
        with tf.variable_scope(name_or_scope=name,reuse=reuse):
            input=tf.reshape(input,[-1,25*50*30])
            out=tf.layers.dense(input,out_size,activation=tf.nn.relu)
        return out

    def build_model(self):
        with tf.device('/device:GPU:%s' % gpu_id):

            self.s_0_emb=self.build_emb(self.sent_0_word,reuse=False)
            self.s_pos_emb=self.build_emb(self.sent_pos_word,reuse=True)
            self.s_neg_emb=self.build_emb(self.sent_neg_word,reuse=True)


            s_0_enc=sent_encoder(self.s_0_emb,self.args.hidden,self.sent_0_len,name="enc",reuse=False)
            s_pos_enc=sent_encoder(self.s_pos_emb,self.args.hidden,self.sent_pos_len,name="enc",reuse=True)
            s_neg_enc=sent_encoder(self.s_neg_emb,self.args.hidden,self.sent_neg_len,name="enc",reuse=True)

            self.s_0_sent_emb=last_relevant_output(s_0_enc,self.sent_0_len)
            self.s_pos_sent_emb=last_relevant_output(s_pos_enc,self.sent_pos_len)
            self.s_neg_sent_emb=last_relevant_output(s_neg_enc,self.sent_neg_len)

            self.distance_pos=tf.sqrt(tf.reduce_sum(tf.square(self.s_0_sent_emb-self.s_pos_sent_emb),axis=1))
            self.distance_neg=tf.sqrt(tf.reduce_sum(tf.square(self.s_0_sent_emb-self.s_neg_sent_emb),axis=1))

            self.cosine_pos=cos_sim(self.s_0_sent_emb,self.s_pos_sent_emb)
            self.cosine_neg=cos_sim(self.s_0_sent_emb,self.s_neg_sent_emb)

            transpose_0_emb = tf.transpose(self.s_0_emb,[0, 1, 2])  # batch_size x sequence_length x words_dimensionality
            transpose_pos_emb = tf.transpose(self.s_pos_emb, [0, 1, 2])
            transpose_neg_emb = tf.transpose(self.s_neg_emb, [0, 1, 2])

            x_0_expanded = tf.expand_dims(transpose_0_emb, -1)
            x_pos_expanded = tf.expand_dims(transpose_pos_emb, -1)
            x_neg_expanded = tf.expand_dims(transpose_neg_emb, -1)


            x_0_cnn_out=self.cnn_encoder_(input=x_0_expanded,name='cnn_enc',reuse=False)
            x_pos_cnn_out=self.cnn_encoder_(input=x_pos_expanded,name='cnn_enc',reuse=True)
            x_neg_cnn_out=self.cnn_encoder_(input=x_neg_expanded,name='cnn_enc',reuse=True)

            # self.s_0_sent_emb=self.full_connect(x_0_cnn_out,self.args.hidden,name='full',reuse=False)
            # self.s_pos_sent_emb=self.full_connect(x_pos_cnn_out,self.args.hidden,name='full',reuse=True)
            # self.s_neg_sent_emb=self.full_connect(x_neg_cnn_out,self.args.hidden,name='full',reuse=True)

            # self.distance_pos = tf.sqrt(tf.reduce_sum(tf.square(self.s_0_sent_emb - self.s_pos_sent_emb),axis=1))
            # self.distance_neg = tf.sqrt(tf.reduce_sum(tf.square(self.s_0_sent_emb - self.s_neg_sent_emb),axis=1))

            # h_1_pos,h_2_pos=self.cnn_encoder(x_0_expanded,x_pos_expanded,reuse_1=False,reuse_2=True)
            # h_1_neg,h_2_neg=self.cnn_encoder(x_0_expanded,x_neg_expanded,reuse_1=True,reuse_2=True)
            #
            #
            #
            # # with tf.variable_scope("feature_layer"):
            #     self.output_features_pos = tf.concat([h_1_pos, h_2_pos, h_1_pos - h_2_pos, h_1_pos * h_2_pos], axis=1, name="output_features_pos")
            #     self.output_features_neg = tf.concat([h_1_neg, h_2_neg, h_1_neg - h_2_neg, h_1_neg * h_2_neg], axis=1, name="output_features_neg")
            #
            # with tf.variable_scope("output_layer"):
            #
            #     self.estimation_pos = tf.contrib.layers.fully_connected(
            #         inputs=self.output_features_pos,
            #         num_outputs=self.args.num_classes,
            #         activation_fn=None,
            #         weights_initializer=tf.contrib.layers.xavier_initializer(),
            #         weights_regularizer=tf.contrib.layers.l2_regularizer(scale=self.args.l2_reg),
            #         biases_initializer=tf.constant_initializer(1e-04),
            #         scope="FC_pos"
            #     )
            #
            #     self.estimation_neg = tf.contrib.layers.fully_connected(
            #         inputs=self.output_features_pos,
            #         num_outputs=self.args.num_classes,
            #         activation_fn=None,
            #         weights_initializer=tf.contrib.layers.xavier_initializer(),
            #         weights_regularizer=tf.contrib.layers.l2_regularizer(scale=self.args.l2_reg),
            #         biases_initializer=tf.constant_initializer(1e-04),
            #         scope="FC_neg"
            #     )

            # self.pred_probs = tf.contrib.layers.softmax(self.estimation)
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
            # self.loss= tf.reduce_mean(tf.nn.relu(self.estimation_pos - self.estimation_neg + self.args.margin))
            loss_distance_triple= tf.maximum(0.0,5.0+self.distance_pos-self.distance_neg)

            loss_consine_triple=tf.maximum(0.0,1.0+self.cosine_pos-self.cosine_neg)

            self.loss=tf.reduce_mean(loss_distance_triple)
            print(self.loss)
            # if self.args.loss == "softmax_loss":
            #     self.loss, _ = point_wise_loss.softmax_loss(self.estimation, self.target,
            #                                                 *args, **kargs)
            # elif self.args.loss == "sparse_amsoftmax_loss":
            #     self.loss, _ = point_wise_loss.sparse_amsoftmax_loss(self.estimation, self.target,
            #                                                          self.args, *args, **kargs)
            # elif self.args.loss == "focal_loss_binary_v2":
            #     self.loss, _ = point_wise_loss.focal_loss_binary_v2(self.estimation, self.target,
            #                                                         self.args, *args, **kargs)
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
            self.sent_0_word: sample_batch["sent_0_word"],
            self.sent_pos_word: sample_batch["sent_pos_word"],
            self.sent_neg_word: sample_batch["sent_neg_word"],
            self.sent_0_len: sample_batch["sent_0_len"],
            self.sent_pos_len: sample_batch["sent_pos_len"],
            self.sent_neg_len: sample_batch["sent_neg_len"],
            self.dpool_index: self.dynamic_pooling_index(np.array(sample_batch["sent_0_len"]), np.array(sample_batch["sent_pos_len"]),
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
                _, loss = sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
            else:
                loss = sess.run(self.loss, feed_dict=feed_dict)
            acc=0.0
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
        print("EP%d_%s, avg_loss=%s  avg_acc=%s "  % (epoch,flag, avg_loss / len(data_loader),avg_acc / len(data_loader)))
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
                self.sent_0_word: data["sent_0_word"],
                self.sent_pos_word: data["sent_pos_word"],
                self.sent_neg_word: data["sent_neg_word"],
                self.sent_0_len: data["sent_0_len"],
                self.sent_pos_len: data["sent_pos_len"],
                self.sent_neg_len: data["sent_neg_len"],

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
        end_time = time.time()


    def get_sent_emb(self,infer_data_loader,restore_model_path,sent_emb_name):
        start_time = time.time()
        self.saver = tf.train.Saver()
        id2vocab=self.args.id2vocab
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        # _logger.info('entity_words:%s'%(entity_dict.keys()))
        if restore_model_path:
            self.saver.restore(self.sess, restore_model_path)
            print("restore model from ", restore_model_path)
        else:
            print('error!!please input restore_model_path')

        sent_emb_dict={}
        for i, data in enumerate(infer_data_loader):
            feed_dict = {
                self.sent_0_word: data["sent_0_word"],
                self.sent_pos_word: data["sent_pos_word"],
                self.sent_neg_word: data["sent_neg_word"],
                self.sent_0_len: data["sent_0_len"],
                self.sent_pos_len: data["sent_pos_len"],
                self.sent_neg_len: data["sent_neg_len"],
            }
            sent_0_emb,sent_pos_emb,sent_neg_emb=self.sess.run([self.s_0_sent_emb,self.s_pos_sent_emb,self.s_neg_sent_emb],feed_dict=feed_dict)

            # cnadidate_list
            sent0_word=data["sent_0_word"]
            for i in range(len(sent0_word)):
                sent = ' '.join([id2vocab[e] for e in np.array(sent0_word[i]) if e != 0])
                sent_emb_dict[sent]=sent_0_emb[i]

            sentpos_word=data["sent_pos_word"]
            for i in range(len(sentpos_word)):
                sent = ' '.join([id2vocab[e] for e in np.array(sentpos_word[i]) if e != 0])
                sent_emb_dict[sent]=sent_pos_emb[i]

            sentneg_word=data["sent_neg_word"]
            for i in range(len(sentneg_word)):
                sent = ' '.join([id2vocab[e] for e in np.array(sentneg_word[i]) if e != 0])
                sent_emb_dict[sent]=sent_neg_emb[i]

        pkl.dump(sent_emb_dict,open('./%s'%sent_emb_name,'wb'))