import numpy as np
import tensorflow as tf
from utils import *
from model.utils.cnn_utils import *
from model.utils.rnn_utils import *
from model.utils.esim import esim_utils
from model.utils.bimpm import layer_utils
from model.utils.slstm import slstm_utils
from model.utils.matchpyramid import rnn_utils,match_utils
from loss import point_wise_loss
import time
import tqdm
import os
gpu_id=0
os.environ["CUDA_VISIBLE_DEVICES"] = "%s"%gpu_id

class EsimModel(object):
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
        self.args.highway_layer_num=2

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

    def build_encoder(self, input_emb, input_lengths,enc_mod='lstm', scope='encoder',reuse=False):

        word_emb = tf.nn.dropout(input_emb, 0.1)
        with tf.variable_scope(scope + "_input_highway", reuse=reuse):
            input_dim = word_emb.get_shape()[-1]
            sent_repres = match_utils.multi_highway_layer(word_emb, input_dim, self.args.highway_layer_num)

            if enc_mod == "lstm":
                [sent_repres_fw, sent_repres_bw, sent_repres] = layer_utils.my_lstm_layer(sent_repres,
                                                                                          self.args.hidden,
                                                                                          input_lengths=input_lengths,
                                                                                          scope_name=scope,
                                                                                          reuse=reuse,
                                                                                          is_training=True,
                                                                                          dropout_rate=0.1,
                                                                                          use_cudnn=False)

            elif enc_mod == "slstm":

                word_emb_proj = tf.layers.dense(word_emb,
                                                self.args.hidden)

                initial_hidden_states = word_emb_proj
                initial_cell_states = tf.identity(initial_hidden_states)

                [new_hidden_states,
                 new_cell_states,
                 dummynode_hidden_states] = slstm_utils.slstm_cell(scope, self.args.hidden,
                                                                   input_lengths,
                                                                   initial_hidden_states,
                                                                   initial_cell_states,
                                                                   2,
                                                                   0.1, reuse=reuse)

                sent_repres = new_hidden_states

        return sent_repres

    def build_interactor(self, sent1_repres, sent2_repres, sent1_len, sent2_len,
                         sent1_mask, sent2_mask, scope='intera',reuse=False,enc_mod='lstm'):

        input_dim = sent1_repres.get_shape()[-1]
        with tf.variable_scope(scope + "_interaction_module", reuse=reuse):
            [query_attention_outputs, context_attention_outputs] = esim_utils.query_context_attention(sent1_repres,
                                                                                                      sent2_repres,
                                                                                                      sent1_len,
                                                                                                      sent2_len,
                                                                                                      sent1_mask,
                                                                                                      sent2_mask,
                                                                                                      0.1,
                                                                                                      scope,
                                                                                                      reuse=reuse)

            if enc_mod == "lstm":
                [sent1_repres_fw, sent1_repres_bw, sent1_repres] = layer_utils.my_lstm_layer(query_attention_outputs,
                                                                                             self.args.hidden,
                                                                                             input_lengths=sent1_len,
                                                                                             scope_name=scope,
                                                                                             reuse=reuse,
                                                                                             is_training=True,
                                                                                             dropout_rate=0.1,
                                                                                             use_cudnn=False)

                [sent2_repres_fw, sent2_repres_bw, sent2_repres] = layer_utils.my_lstm_layer(context_attention_outputs,
                                                                                             self.args.hidden,
                                                                                             input_lengths=sent2_len,
                                                                                             scope_name=scope,
                                                                                             reuse=True,
                                                                                             is_training=True,
                                                                                             dropout_rate=0.1,
                                                                                             use_cudnn=False)
                match_dim = self.args.hidden * 8

            elif enc_mod == "slstm":

                sent1_initial_hidden_states = tf.layers.dense(query_attention_outputs,
                                                              self.args.hidden)
                sent1_initial_cell_states = tf.identity(sent1_initial_hidden_states)

                [new_sent1_hidden_states,
                 new_sent1_cell_states,
                 dummynode_sent1_hidden_states] = slstm_utils.slstm_cell(
                                                                         scope,
                                                                         self.args.hidden,
                                                                         sent1_len,
                                                                         sent1_initial_hidden_states,
                                                                         sent1_initial_cell_states,
                                                                         2,
                                                                         0.1, reuse=False)

                sent1_repres = new_sent1_hidden_states

                sent2_initial_hidden_states = tf.layers.dense(context_attention_outputs,
                                                              self.args.hidden)
                sent2_initial_cell_states = tf.identity(sent2_initial_hidden_states)

                [new_sent2_hidden_states,
                 new_sent2_cell_states,
                 dummynode_sent2_hidden_states] = slstm_utils.slstm_cell(
                                                                         scope,
                                                                         self.args.hidden,
                                                                         sent2_len,
                                                                         sent2_initial_hidden_states,
                                                                         sent2_initial_cell_states,
                                                                         2,
                                                                         0.1, reuse=True)

                sent2_repres = new_sent2_hidden_states
                match_dim = self.args.hidden * 4

            v_1_sum = tf.reduce_sum(sent1_repres, 1)
            v_1_ave = tf.div(v_1_sum, tf.expand_dims(tf.cast(sent1_len, tf.float32) + EPSILON, -1))

            v_2_sum = tf.reduce_sum(sent2_repres, 1)
            v_2_ave = tf.div(v_2_sum, tf.expand_dims(tf.cast(sent2_len, tf.float32) + EPSILON, -1))

            v_1_max = tf.reduce_max(sent1_repres, 1)
            v_2_max = tf.reduce_max(sent2_repres, 1)

            out1 = tf.concat([v_1_ave, v_1_max], axis=-1)
            out2 = tf.concat([v_2_ave, v_2_max], axis=-1)

            out = tf.concat([v_1_ave, v_2_ave, v_1_max, v_2_max], 1)

            return out1, out2, out, match_dim

    def build_model(self):
        with tf.device('/device:GPU:%s' % gpu_id):

            s1_emb=self.build_emb(self.sent1_word,reuse=False)
            s2_emb=self.build_emb(self.sent2_word,reuse=True)

            self.sent1_encoded = self.build_encoder(s1_emb,
                                                    self.sent1_len,
                                                    reuse=False)

            self.sent2_encoded = self.build_encoder(s2_emb,
                                                    self.sent2_len,
                                                    reuse=True)

            [self.sent1_repres,
             self.sent2_repres,
             self.aggregat_repres,
             match_dim] = self.build_interactor(self.sent1_encoded,
                                                self.sent2_encoded,
                                                self.sent1_len,
                                                self.sent2_len,
                                                self.sent1_token_mask,
                                                self.sent2_token_mask)

        with tf.variable_scope(self.scope + "_prediction_module",):
            # ========Prediction Layer=========
            # match_dim = 4 * self.options.aggregation_lstm_dim
            w_0 = tf.get_variable("w_0", [match_dim, match_dim / 2], dtype=tf.float32)
            b_0 = tf.get_variable("b_0", [match_dim / 2], dtype=tf.float32)
            w_1 = tf.get_variable("w_1", [match_dim / 2, self.args.num_classes], dtype=tf.float32)
            b_1 = tf.get_variable("b_1", [self.args.num_classes], dtype=tf.float32)

            # if is_training: match_representation = tf.nn.dropout(match_representation, (1 - options.dropout_rate))
            logits = tf.matmul(self.aggregat_repres, w_0) + b_0
            logits = tf.tanh(logits)
            logits = tf.nn.dropout(logits, (0.1))

            self.logits = tf.matmul(logits, w_1) + b_1
            self.pred_probs = tf.nn.softmax(self.logits)


    def build_accuracy(self, *args, **kargs):
        self.pred_label = tf.arg_max(self.pred_probs,-1)

        correct = tf.equal(
            tf.cast(self.pred_label, tf.int32),
            tf.cast(self.target, tf.int32)
        )
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    def build_loss(self, *args, **kargs):
        with tf.device('/device:GPU:%s' % gpu_id):
            if self.args.loss == "softmax_loss":
                # self.loss=self.loss
                self.loss, _ = point_wise_loss.softmax_loss(self.logits, self.target,
                                                            *args, **kargs)
            elif self.args.loss == "sparse_amsoftmax_loss":
                # self.loss=tf.reduce_mean(self.distance_loss)
                #
                self.loss, _ = point_wise_loss.sparse_amsoftmax_loss(self.logits, self.target,
                                                                     self.args, *args, **kargs)
            elif self.args.loss == "focal_loss_binary_v2":
                # self.loss=tf.reduce_mean(self.distance_loss)
                #
                self.loss, _ = point_wise_loss.focal_loss_binary_v2(self.logits, self.target,
                                                                    self.args, *args, **kargs)
    def build_op(self):
        with tf.device('/device:GPU:%s' % gpu_id):
            if self.args.opt == 'adadelta':
                self.optimizer = tf.train.AdadeltaOptimizer(self.args.lr).minimize(self.loss)
            elif self.args.opt == 'adam':
                # self.optimizer = tf.train.AdamOptimizer(0.3)
                # grad_var=self.optimizer.compute_gradients(self.loss)
                # for g,v in grad_var:
                #     print(g,v)
                # # cappd_grad_varible=[[tf.clip_by_value(g,1e-5,1.0),v] for g,v in grad_var]
                # # optimizer=opt.apply_gradients(grads_and_vars=cappd_grad_varible)
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

        fw=open('./write_%s.txt'%self.scope,'w',encoding='utf-8')
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
                cand_label = self.args.id2label[infer_pred_label[i]]

                fw.write(sent1+'\t\t'+sent2+'\t\t'+str(prob)+'\t\t'+str(cand_label)+'\n')
                # print(sent1, '###', sent2, "###", prob, '###', cand_label, '\n')

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