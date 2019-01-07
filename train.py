from model.BimpmModel import Bimpm
from model.MatchPyramid import MatchPyramid
from model.LstmMatchPyramid import LstmMatchPyramid
from model.Transformer import TransformerModel
from model.SiameseCNN import SiameseCNN
from model.SiameseLstm import SiameseLstm
from model.ABCNN import ABCNN
from model.DSSM import Dssm
from model.ARCI import ARC_I
from model.LstmMatchPyramid import LstmMatchPyramid
from model.EsimModel import EsimModel
from dataset import BERTDataset,WordVocab,BERTDatasetFineTune,DatasetCandidate,DatasetPairs
from dataset.dataset_tfrecord import DataSetTfrecord
import tensorflow as tf
import pickle as pkl
import json
import numpy as np
from urllib import parse
from urllib import request
import argparse
from data.utils import tools

import ast
from torch.utils.data import DataLoader
import os
PATH=os.path.split(os.path.realpath(__file__))[0]
seq_len=50
import logging
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
logger.addHandler(ch)
_logger=logger
def build_vocab(corpus_path,vocab_path):
    with open(corpus_path, "r",encoding='utf-8') as f:
        vocab = WordVocab(f, max_size=None, min_freq=1)

        print("VOCAB SIZE:", len(vocab))
        vocab.save_vocab(vocab_path)

def build_label_vocab(train_corpus,test_corpus,label_vocab_path):
    label_dict={}
    index=len(label_dict)
    for ele in open(train_corpus,'r',encoding='utf-8'):
        label=ele.replace('\n','').split('\t')[-1]
        if label not in label_dict:
            label_dict[label]=index
            index+=1
    for ele in open(test_corpus,'r',encoding='utf-8'):
        label=ele.replace('\n','').split('\t')[-1]
        if label not in label_dict:
            label_dict[label]=index
            index+=1

    pkl.dump(label_dict,open(label_vocab_path,'wb'))
    return label_dict


def read_tfRecord(seq_len,batch_size):
    def _read_tfrecoder(file_tfRecord):
        # queue = tf.train.string_input_producer([file_tfRecord])
        # reader = tf.TFRecordReader()
        # _,serialized_example = reader.read(queue)

        filename_queue = tf.train.string_input_producer(
            tf.train.match_filenames_once(file_tfRecord),
            shuffle=True, num_epochs=None)  # None表示没哟限制
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(
                serialized_example,
                features={
              'sent1_word': tf.FixedLenFeature([seq_len], tf.int64),
              'sent2_word': tf.FixedLenFeature([seq_len], tf.int64),
              'sent1_len':tf.FixedLenFeature([1], tf.int64),
              'sent2_len': tf.FixedLenFeature([1], tf.int64),
              'label': tf.FixedLenFeature([1], tf.int64)
                        }
                )
        sent1_word = tf.cast(features['sent1_word'], tf.int32)
        sent2_word = tf.cast(features['sent2_word'], tf.int32)
        sent1_len = tf.cast(features['sent1_len'], tf.int32)
        sent2_len = tf.cast(features['sent2_len'], tf.int32)
        label= tf.cast(features['label'], tf.int32)

        sent1_word=tf.reshape(sent1_word,[-1,seq_len])
        sent2_word=tf.reshape(sent2_word,[-1,seq_len])

        # input_queue=[sent1_word, sent2_word,sent1_len,sent2_len,label]
        input_queue = tf.train.slice_input_producer([sent1_word, sent2_word,sent1_len,sent2_len,label], shuffle=False)
        sent1_word_batch, sent2_word_batch, sent1_len_batch, sent2_len_batch, label_batch \
            = tf.train.batch(input_queue,batch_size=batch_size,allow_smaller_final_batch=True,num_threads=4)

        # with tf.Session() as sess:
        #     tf.global_variables_initializer().run()
        #     tf.local_variables_initializer().run()
        #
        #     coord = tf.train.Coordinator()
        #     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        #
        #     for i in range(1):
        #         s,s1 = sess.run([sent1_word_batch, sent2_len_batch])
        #         print('batch' + str(i) + ':')
        #         print(s.shape)
        #         print(s1.shape)
        #         # print('width:%d \n', wid[i])
        #
        #     coord.request_stop()
        #     coord.join(threads)

        return [sent1_word_batch, sent2_word_batch, sent1_len_batch, sent2_len_batch, label_batch]
    return _read_tfrecoder

def embedding_api(word):
    origin_word=word
    word = parse.quote(word)
    with request.urlopen('http://kfdata.cm.com/interface/embedding/getEmbedding?key=%s' % word) as f:
        data = f.read()
        res=json.loads(data.decode("utf-8"))['msg']
        state=json.loads(data.decode("utf-8"))['rs']
        res_emb=list(np.random.random((200,)))
        try:
            res_emb=[float(e) for e in res.split(',')]
        except:
            print('error:%s'%origin_word)
        # print(res_emb)
        return state,res_emb

def getEmbeddimhArray(vocab):
    emb_list=[]
    for k,v in vocab.stoi.items():
        state,emb=embedding_api(k)
        emb_list.append(emb)
    return np.array(emb_list,dtype=np.float32)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_path",
                        help="vocab path", type=str, default='./data/vocab_word.big')#vocab_8kw
    parser.add_argument("--label_vocab_path",
                        help="vocab label_vocab_path", type=str, default='./data/label_vocab.big')
    parser.add_argument("--train_dataset",
                        help="train_dataset", type=str, default='./data/sim_train_word_small.txt')
    parser.add_argument("--test_dataset",
                        help="test_dataset", type=str, default='./data/sim_test_word_small.txt')
    parser.add_argument("--output_path",
                        help="output_path", type=str, default='./output')
    parser.add_argument("--seq_len",
                        help="seq_len", type=int, default=50)
    parser.add_argument("--batch_size",
                        help="batch_size", type=int, default=256)
    parser.add_argument("--num_workers",
                        help="num_workers", type=int, default=6)
    parser.add_argument("--emb_size",
                        help="emb_size", type=int, default=200)
    parser.add_argument("--hidden",
                        help="hidden", type=int, default=256)
    parser.add_argument("--highway_layer_num",
                        help="highway_layer_num", type=int, default=1)
    parser.add_argument("--layers",
                        help="layers", type=int, default=4)
    parser.add_argument("--attn_heads",
                        help="attn_heads", type=int, default=4)
    parser.add_argument("--lr",
                        help="lr", type=float, default=0.001)
    parser.add_argument("--dropout",
                        help="dropout", type=float, default=0.0)
    parser.add_argument("--adam_beta1",
                        help="adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2",
                        help="adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay",
                        help="adam_weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs",
                        help="epochs", type=int, default=200)
    parser.add_argument("--sinusoid",
                        help="sinusoid", type=ast.literal_eval, default=False)
    parser.add_argument("--loss",
                        help="loss", type=str, default='softmax_loss')
    parser.add_argument("--opt",
                        help="opt", type=str, default='adam')
    parser.add_argument("--use_tfrecord",
                        help="use_tfrecord", type=ast.literal_eval, default=False)
    parser.add_argument("--train_tfrecord_path",
                        help="train_tfrecord_path", type=str, default='./train_2kw.tfrecord')
    parser.add_argument("--test_tfrecord_path",
                        help="test_tfrecord_path", type=str, default='./test_2kw.tfrecord')

    parser.add_argument("--model_type",
                        help="abcnn's attention model ABCNN1/ABCNN2/ABCNN3/BCNN",
                        type=str, default='BCNN')
    parser.add_argument("--new_vocab",
                        help="new_vocab", type=ast.literal_eval, default=True)
    parser.add_argument("--new_tfrecord",
                        help="new_tfrecord", type=ast.literal_eval, default=False)
    parser.add_argument("--model_name",
                        help="model_name", type=str, default='siamese_cnn')
    parser.add_argument("--corpus_lines",
                        help="corpus_lines", type=int, default=1)
    parser.add_argument("--task_name",help="task name", type=str, default="QQP")
    parser.add_argument("--restore_model",help="", type=str, default="")
    parser.add_argument("--use_pre_train_emb",help="", type=ast.literal_eval, default=False)
    parser.add_argument("--do_train",help="", type=ast.literal_eval, default=False)
    parser.add_argument("--do_predict",help="", type=ast.literal_eval, default=False)
    parser.add_argument("--do_pairs_predict",help="", type=ast.literal_eval, default=False)
    parser.add_argument("--cosine",help="", type=ast.literal_eval, default=False)

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    # _logger.info("build label vocab")
    # label_vocab=build_label_vocab(args.train_dataset,args.test_dataset,args.label_vocab_path)
    # _logger.info(label_vocab)
    # pkl.dump(label_vocab,open(PATH+"/label_vocab.p",'wb'))
    # _logger.info("label vocab size:%s"%(len(label_vocab)))
    _logger.info("new_vocab:%s"%args.new_vocab)
    _logger.info("use_tfrecord:%s"%args.use_tfrecord)
    _logger.info("train_dataset:%s"%args.train_dataset)
    _logger.info("test_dataset:%s"%args.test_dataset)
    _logger.info("task_name:%s"%args.task_name)
    _logger.info("model_name:%s"%args.model_name)
    _logger.info("new_tfrecord:%s"%args.new_tfrecord)
    _logger.info("restore_model:%s"%args.restore_model)
    _logger.info("use_pre_train_emb:%s"%args.use_pre_train_emb)

    label_vocab={"0":0,'1':1}
    if not args.new_vocab and os.path.exists(args.vocab_path):
        _logger.info("Loading Vocab: %s"%(args.vocab_path))
        vocab = WordVocab.load_vocab(args.vocab_path)
    else:
        _logger.info("build vocab")
        build_vocab(args.train_dataset,args.vocab_path)
        _logger.info("Loading Vocab: %s" % (args.vocab_path))
        vocab = WordVocab.load_vocab(args.vocab_path)
    _logger.info("Vocab Size:%s"%(len(vocab)))
    args.vocab_size=len(vocab)
    args.num_classes = len(label_vocab)
    if args.use_pre_train_emb:
        if args.new_vocab:
            vocab_emb=getEmbeddimhArray(vocab)
            pkl.dump(vocab_emb,open('vocab_emb.p','wb'))
        else:
            vocab_emb=pkl.load(open('vocab_emb.p','rb'))
        args.vocab_emb=vocab_emb
        _logger.info('load pre_train_emb finish emb_array size:%s'%(len(vocab_emb)))

    if args.use_tfrecord:
        if not os.path.exists(args.train_tfrecord_path) or not os.path.exists(args.test_tfrecord_path) or args.new_tfrecord:
            _logger.info('building tfrecords')
            DataSetTfrecord(args.train_dataset, vocab, args.seq_len, corpus_lines=args.corpus_lines,
                            label_vocab=label_vocab, out_path=args.train_tfrecord_path)
            DataSetTfrecord(args.test_dataset, vocab, args.seq_len, corpus_lines=args.corpus_lines, label_vocab=label_vocab,
                            out_path=args.test_tfrecord_path)
        _read_tfRecord=read_tfRecord(args.seq_len,args.batch_size)
        _logger.info("loading tfrecords")
        train_data_loader=_read_tfRecord(args.train_tfrecord_path)
        test_data_loader=_read_tfRecord(args.test_tfrecord_path)
        train_num=int([e for e in open(args.train_tfrecord_path+".index",'r',encoding='utf-8').readlines()][0])
        test_num=int([e for e in open(args.test_tfrecord_path+".index",'r',encoding='utf-8').readlines()][0])
        _logger.info('train_num:%s  test_num:%s'%(train_num,test_num))
        args.train_num=train_num
        args.test_num=test_num

    else:
        _logger.info('loading dataset')
        train_dataset = BERTDatasetFineTune(args.train_dataset, vocab, seq_len=args.seq_len, corpus_lines=args.corpus_lines,label_vocab=label_vocab)
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        test_dataset = BERTDatasetFineTune(args.test_dataset, vocab,
                               seq_len=args.seq_len,label_vocab=label_vocab) if args.test_dataset is not None else None
        test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) if test_dataset is not None else None
    _logger.info('%s  batch_size:%s  use_tfrecod:%s'%(args.model_name,args.batch_size,args.use_tfrecord))
    model_name=args.model_name
    if model_name == 'bimpm':
        model=Bimpm(args,'bimpm')
    elif model_name == 'mp':
        model=MatchPyramid(args,'matchpyramid')
    elif model_name == 'lstm_mp':
        model = LstmMatchPyramid(args, "lstm_matchpyramid")
    elif model_name == 'siamese_cnn':
        model = SiameseCNN(args,"SiameseCNN")
    elif model_name == 'siamese_lstm':
        model = SiameseLstm(args,"SiameseLstm")
    elif model_name == 'transformer':
        model = TransformerModel(args,"Transformer")
    elif model_name == 'esim':
        model=EsimModel(args,"Esim")
    elif model_name == 'arli':
        model=ARC_I(args,"ArlI")
    elif model_name == 'dssm':
        model=Dssm(args,"dssm")
    elif model_name == 'abcnn':
        model=ABCNN(args,"abcnn")

    model.build_placeholder()
    model.build_model()
    model.build_accuracy()
    model.build_loss()
    model.build_op()

    # emb=model.getEmbedding()

    if args.do_train:
        if args.restore_model=='':
            model.train(train_data_loader, test_data_loader,restore_model=None, save_model=PATH + "/output/%s_%s_2kw.ckpt"%(model_name,args.task_name))
        else:
            model.train(train_data_loader, test_data_loader,restore_model=PATH+args.restore_model, save_model=PATH + "/output/%s_%s_2kw.ckpt"%(model_name,args.task_name))

    if args.do_predict:
        candidata_list_origin = ["怎 么 解 除 num 天 红包 限 制"]
        infer_list_origin = ["能 帮 我 找 回 吗",]
        candidata_list = []
        infer_list = []
        tool=tools()
        for sent in candidata_list_origin:
            sent=sent.strip()
            sent = tool.pre_deal_sent(sent, type='char')
            candidata_list.append(sent)

        for sent in infer_list_origin:
            sent=sent.strip()
            sent = tool.pre_deal_sent(sent, type='char')
            infer_list.append(sent)

        print("loading label dict")
        label_dict = {"0": 0, '1': 1}
        args.id2label = {v: k for k, v in label_dict.items()}
        print("Loading Vocab", args.vocab_path)
        vocab = WordVocab.load_vocab(args.vocab_path)
        id2word = {v: k for k, v in vocab.stoi.items()}
        print(id2word)
        args.id2word = id2word
        print("Vocab Size: ", len(vocab))
        args.vocab_size = len(vocab)

        print("Loading test Dataset", args.test_dataset)
        infer_data = DatasetCandidate(args.test_dataset, vocab, seq_len=args.seq_len, candidate_dict=None,
                                      label_dict=label_dict, candidata_list=candidata_list, infer_list=infer_list)
        print(infer_data.__len__())
        print("Creating Dataloader")
        infer_data_loader = DataLoader(infer_data, batch_size=args.batch_size, num_workers=args.num_workers)
        print(len(infer_data_loader))

        model.infer(infer_data_loader, restore_model_path=PATH + "/output/%s_%s_2kw.ckpt"%(model_name,args.task_name))

    if args.do_pairs_predict:
        data_list_origin = [["怎 么 解 除 num 天 红包 限 制","能 帮 我 找 回 吗"],  #0
                            ["我不知道这个红包发给谁了","我要的是红包发给哪个具体的人"],  #1
                            ["人机练习模式都有什么模式","红包发了三次 对方只收到一次"],  #0
                            ["对 方 已 经 领 了","拿 了 还 能 不 能 退 啊"],  #1
                            ["没有收到人脸识别怎么整","提示人脸识别怎么整"],  #0
                            ["要我人脸识别呀", "我该怎么人脸识别呀"],  # 1
                            ["想改限额","银行卡被盗刷"],#0
                            ["有困难人机模式吗","一般电脑模式"],#1
                            ]
        data_list=[]
        tool = tools()
        for sents in data_list_origin:
            sent1 = sents[0].strip()
            sent2 = sents[1].strip()
            sent1 = tool.pre_deal_sent(sent1, type='char')
            sent2 = tool.pre_deal_sent(sent2, type='char')
            data_list.append([sent1,sent2])

        print("loading label dict")
        label_dict = {"0": 0, '1': 1}
        args.id2label = {v: k for k, v in label_dict.items()}
        print("Loading Vocab", args.vocab_path)
        vocab = WordVocab.load_vocab(args.vocab_path)
        id2word = {v: k for k, v in vocab.stoi.items()}
        print(id2word)
        args.id2word = id2word
        print("Vocab Size: ", len(vocab))
        args.vocab_size = len(vocab)

        print("Loading test Dataset", args.test_dataset)
        infer_data = DatasetPairs(vocab=vocab, seq_len=args.seq_len, label_dict=label_dict,data_pairs_list=data_list)
        print(infer_data.__len__())
        print("Creating Dataloader")
        infer_data_loader = DataLoader(infer_data, batch_size=args.batch_size, num_workers=args.num_workers)
        print(len(infer_data_loader))

        model.infer(infer_data_loader,
                    restore_model_path=PATH + "/output/%s_%s_2kw.ckpt" % (model_name, args.task_name))
if __name__ == '__main__':
    train()