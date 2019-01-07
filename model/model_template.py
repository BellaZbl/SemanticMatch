import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod
from model.utils.embed import integration_func
import os
from loss import point_wise_loss
gpu_id=0
os.environ["CUDA_VISIBLE_DEVICES"] = "%s"%gpu_id
import logging
from collections import defaultdict
import time
import tqdm
PATH=os.path.split(os.path.realpath(__file__))[0]
logger=logging.getLogger()
logger.setLevel(logging.INFO)
logfile=PATH+'/log/sematic_match.txt'
fh=logging.FileHandler(logfile,mode='w')
fh.setLevel(logging.DEBUG)
ch=logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter=logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
_logger=logger

class ModelTemplate(object):
    __metaclass__ = ABCMeta
    def __init__(self,args,scope):
        self.args=args
        self.scope=scope
        self.gpu_id=gpu_id
    @abstractmethod
    def build_placeholder(self, ):
        pass

    @abstractmethod
    def build_accuracy(self, *args, **kargs):
        self.pred_label = tf.argmax(self.pred_probs, axis=-1)
        correct = tf.equal(
            tf.cast(self.pred_label, tf.int32),
            tf.cast(self.target, tf.int32)
        )
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    def embedding(self,inputs,vocab_size,num_units,zero_pad=False,scale=True,scope="embedding",reuse=False):
       '''
       :param vocab_size:
       :param num_units:
       :param zero_pad:
       :param scale:
       :param scope:
       :param reuse:
       :return:
       '''
       with tf.variable_scope(scope, reuse=reuse):
           if self.args.use_pre_train_emb:
               assert  vocab_size==self.args.vocab_emb.shape[0]
               lookup_table = tf.get_variable('lookup_table',
                                              dtype=tf.float32,
                                              shape=[vocab_size, num_units],
                                              trainable=False,
                                              initializer=tf.constant_initializer(self.args.vocab_emb, dtype=tf.float32))
           else:
                lookup_table = tf.get_variable('lookup_table',
                                               dtype=tf.float32,
                                               shape=[vocab_size, num_units],
                                               trainable=True,
                                               initializer=tf.contrib.layers.xavier_initializer())
           if zero_pad:
                lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),lookup_table[1:, :]), 0)
           outputs = tf.nn.embedding_lookup(lookup_table, inputs)
           if scale:
                outputs = outputs * (num_units ** 0.5)

           self.EmbeddingTable=lookup_table
       return outputs

    @abstractmethod
    def build_loss(self, *args, **kargs):
        with tf.device('/device:GPU:%s' % gpu_id):
            if self.args.loss == "softmax_loss":
                # log_probs = tf.log(self.estimation)
                self.soft_loss, _ = point_wise_loss.softmax_loss(self.estimation, self.target,
                                                            *args, **kargs)

                self.loss=self.soft_loss
            elif self.args.loss == "sparse_amsoftmax_loss":
                self.loss, _ = point_wise_loss.sparse_amsoftmax_loss(self.estimation, self.target,
                                                                     self.args, *args, **kargs)
            elif self.args.loss == "focal_loss_binary_v2":
                self.loss, _ = point_wise_loss.focal_loss_binary_v2(self.estimation, self.target,
                                                                  self.args, *args, **kargs)

    @abstractmethod
    def build_op(self):
        with tf.device('/device:GPU:%s' % self.gpu_id):
            if self.args.opt == 'adadelta':
                self.optimizer = tf.train.AdadeltaOptimizer(self.args.lr).minimize(self.loss)
            elif self.args.opt == 'adam':
                self.optimizer = tf.train.AdamOptimizer(self.args.lr).minimize(self.loss)
            elif self.args.opt == 'rmsprop':
                self.optimizer = tf.train.RMSPropOptimizer(self.args.lr).minimize(self.loss)

    @abstractmethod
    def get_feed_dict(self, sample_batch, dropout_keep_prob=None, data_type='train'):

        feed_dict = {
            self.sent1_word: sample_batch["sent1_word"],
            self.sent2_word: sample_batch["sent2_word"],
            self.sent1_len: sample_batch["sent1_len"],
            self.sent2_len: sample_batch["sent2_len"],
            self.target: sample_batch["label"],
        }
        if self.scope in ['matchpyramid']:
            feed_dict.update({self.dpool_index: self.dynamic_pooling_index(np.array(sample_batch["sent1_len"]), np.array(sample_batch["sent2_len"]),
                                                         self.args.seq_len, self.args.seq_len),})
        return feed_dict


    def getEmbedding(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            _logger.info("variables_initializer")
            sess.run(tf.global_variables_initializer())

            return sess.run(self.EmbeddingTable)

    def iteration(self, sess,epoch, data_loader, train=True):
        """

        :param epoch:
        :param data_loader:
        :param train:
        :return:
        """
        if train:
            flag='train'
        else:
            flag='test'

        if self.args.use_tfrecord:
            if train:
                flag = 'train'
                batch_num = int(self.args.train_num / self.args.batch_size)
                droput=self.args.dropout
            else:
                flag = 'test'
                droput=0.0
                batch_num = int(self.args.test_num / self.args.batch_size)
            sent1_word,sent2_word,sent1_len,sent2_len,label=\
                data_loader[:]
            pbar = tqdm.tqdm(total=batch_num)
            avg_loss, total_correct, total_element, avg_acc = 0.0, 0.0, 0.0, 0.0

            for i in range(batch_num):
                sent1_word_,sent2_word_,sent1_len_,sent2_len_,label_=sess.run([sent1_word,sent2_word,sent1_len,sent2_len,label])
                feed_dict={
                    self.sent1_word: sent1_word_,
                    self.sent2_word: sent2_word_,
                    self.sent1_len: sent1_len_,
                    self.sent2_len: sent2_len_,
                    self.target: label_,
                    self.dropout:droput
                }
                if self.scope in ['matchpyramid']:
                    feed_dict.update({self.dpool_index: self.dynamic_pooling_index(np.array(sent1_len_),
                                                                                   np.array(sent2_len_),
                                                                                   self.args.seq_len,
                                                                                   self.args.seq_len), })
                if train:
                    _, loss, acc = sess.run([self.optimizer, self.loss, self.accuracy], feed_dict=feed_dict)
                else:
                    loss, acc = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
                avg_loss += loss
                total_correct += 0
                total_element += 1
                avg_acc += acc
                pbar.update(i)
            _logger.info("model_name:%s  EP%d_%s, avg_loss=%s  avg_acc=%s" % (self.scope,
            epoch, flag, avg_loss / batch_num, avg_acc / batch_num))
            print('\n')
            pbar.close()

        else:
            pbar = tqdm.tqdm(total=len(data_loader))
            avg_loss,total_correct,total_element,avg_acc=0.0,0.0,0.0,0.0
            for i, data in enumerate(data_loader):
                data = {key: value for key, value in data.items()}
                feed_dict = self.get_feed_dict(data)

                if train:
                    feed_dict.update({self.dropout: self.args.dropout})
                    _, loss,acc = sess.run([self.optimizer, self.loss,self.accuracy], feed_dict=feed_dict)
                else:
                    feed_dict.update({self.dropout: self.args.dropout})
                    loss,acc = sess.run([self.loss,self.accuracy], feed_dict=feed_dict)
                avg_loss += loss
                total_correct += 0
                total_element += 1
                avg_acc+=acc
                pbar.update(i)
            _logger.info("EP%d_%s, avg_loss=%s  avg_acc=%s"  % (epoch,flag, avg_loss / len(data_loader),avg_acc / len(data_loader)))
            pbar.close()

    def dynamic_pooling_index(self, len1, len2, max_len1, max_len2):
        def dpool_index_(batch_idx, len1_one, len2_one, max_len1, max_len2):
            stride1 = 1.0 * max_len1 / len1_one #len1_one sent1 real len
            stride2 = 1.0 * max_len2 / len2_one
            idx1_one = np.arange(max_len1)/stride1
            idx2_one = np.arange(max_len2)/stride2
            mesh1, mesh2 = np.meshgrid(idx1_one, idx2_one)
            index_one = np.transpose(np.stack([np.ones(mesh1.shape) * batch_idx, mesh1, mesh2]), (2, 1, 0))
            return index_one
        index = np.zeros((len(len1), max_len1, max_len2, 3), dtype=int)
        for i in range(len(len1)):
            index[i] = dpool_index_(i, len1[i], len2[i], max_len1, max_len2)
        return index


    def train(self,train_data_loader,test_data_loader=None,restore_model=None,save_model=None):
        start_time=time.time()
        saver=tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True)

        with tf.Session(config=config) as sess:
            if restore_model:
                saver.restore(sess,restore_model)
                _logger.info("restore model from :%s "%(restore_model))
            else:
                _logger.info("variables_initializer")
                sess.run(tf.global_variables_initializer())
            tf.local_variables_initializer().run()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            for epoch in range(self.args.epochs):
                self.iteration(sess,epoch,train_data_loader,True)
                self.iteration(sess,epoch,test_data_loader,False)
                _logger.info('\n\n')
                if save_model:
                    self.save(saver,sess,save_model)
            end_time=time.time()
            coord.request_stop()
            coord.join(threads)
            print('#'*20,end_time-start_time)

    def save(self,saver,sess,save_model):
        saver.save(sess,save_model)
        _logger.info('save in :%s'%(save_model))

    def infer(self,infer_data_loader,restore_model_path):
        start_time = time.time()
        self.saver = tf.train.Saver()
        id2word=self.args.id2word
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        # _logger.info('entity_words:%s'%(entity_dict.keys()))
        if restore_model_path:
            self.saver.restore(self.sess, restore_model_path)
            _logger.info("restore model from :%s"%(restore_model_path))
        else:
            _logger.info('error!!please input restore_model_path')

        fw=open('./write_%s.txt'%self.scope,'w',encoding='utf-8')
        for i, data in enumerate(infer_data_loader):
            feed_dict = {
                self.sent1_word: data["sent1_word"],
                self.sent2_word: data["sent2_word"],
                self.sent1_len: data["sent1_len"],
                self.sent2_len: data["sent2_len"],
                self.dropout:0.0,
            }
            if self.scope in ['matchpyramid']:
                feed_dict.update({self.dpool_index: self.dynamic_pooling_index(["sent1_len"],
                                                                               data["sent2_len"],
                                                                               self.args.seq_len,
                                                                               self.args.seq_len), })

            infer_pred_label,infer_pred_probs=self.sess.run([self.pred_label,self.pred_probs],feed_dict=feed_dict)
            # cnadidate_list
            sent2_word=data["sent2_word"]
            sent1_word=data["sent1_word"]

            for i in range(len(data["sent1_word"])):
                sent1 = ' '.join([id2word[e] for e in np.array(sent1_word[i]) if e != 0])
                sent2 = ' '.join([id2word[e] for e in np.array(sent2_word[i]) if e != 0])
                prob = infer_pred_probs[i]
                cand_label = self.args.id2label[infer_pred_label[i]]

                fw.write(sent1+'\t\t'+sent2+'\t\t'+str(prob)+'\t\t'+str(cand_label)+'\n')
                print(sent1, '###', sent2, "###", prob, '###', cand_label, '\n')

            # infer_pred_label=1-infer_pred_label
            # s=infer_pred_label*np.array(data['candidate_label'])
            # s_prb=infer_pred_label*np.max(infer_pred_probs,1)
            # s_index=np.where(s_prb>0.5)[0]
            # sent2_word=data["sent2_word"]
            # sent1_word=data["sent1_word"]
            # for i in list(s_index):
            #     sent1=' '.join([id2word[e] for e in np.array(sent1_word[i]) if e !=0])
            #     sent2=' '.join([id2word[e] for e in np.array(sent2_word[i]) if e !=0])
            #     prob=s_prb[i]
            #     cand_label=s[i]
            #     true_label=data['true_label'][i]
            #     print(sent1,'###',sent2,"###",prob,'###',cand_label,"###",true_label,'\n')\



            # print(s)
        end_time = time.time()

        print('#' * 20, end_time - start_time)



    def infer_(self,infer_data_loader,restore_model_path):
        start_time = time.time()
        self.saver = tf.train.Saver()
        id2word=self.args.id2word
        id2label=self.args.id2label
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        # _logger.info('entity_words:%s'%(entity_dict.keys()))
        if restore_model_path:
            self.saver.restore(self.sess, restore_model_path)
            print("restore model from ", restore_model_path)
        else:
            print('error!!please input restore_model_path')

        sent_dict=defaultdict(list)
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
            label=data['label'].numpy()
            for i in range(len(data["sent1_word"])):
                sent1 = ' '.join([id2word[e] for e in np.array(sent1_word[i]) if e != 0])
                sent2 = ' '.join([id2word[e] for e in np.array(sent2_word[i]) if e != 0])
                label_=id2label[label[i]]

                prob = infer_pred_probs[i]
                cand_label = infer_pred_label[i]
                if cand_label in [1,"1"]:
                    sent_dict[sent1].append(label_)

        return sent_dict
