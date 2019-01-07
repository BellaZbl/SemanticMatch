from torch.utils.data import Dataset
import tqdm
import torch
import random
import tensorflow as tf
import numpy as np
import os
import collections



class DataSetTfrecord(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None,label_vocab=None,out_path=None
                 ):
        self.vocab = vocab
        self.seq_len = seq_len
        self.label_vocab=label_vocab
        self.corpus_path=corpus_path
        self.trans2tfRecord(output_path=out_path)


    def trans2tfRecord(self,output_path=None):

        filename = output_path
        writer = tf.python_io.TFRecordWriter(filename)
        writer_index=open(output_path+".index",'w',encoding='utf-8')
        # lines = np.genfromtxt(self.corpus_path, delimiter="\t\t", dtype=str, encoding='utf-8')
        index=0
        fr=open(self.corpus_path,'r',encoding='utf-8')
        # for t1, t2, label_ in lines:
        # for line in fr.readlines():
        with open(self.corpus_path,'r',encoding='utf-8') as fr:
            for line in fr:
                t1, t2, label_=line.replace('\n','').split('\t\t')
                if label_ not in self.label_vocab:
                    label = 0
                else:
                    label = self.label_vocab[label_]
                # [CLS] tag = SOS tag, [SEP] tag = EOS tag
                t1 = self.vocab.to_seq(t1)
                t2 = self.vocab.to_seq(t2)
                t1 = [self.vocab.sos_index] + t1 + [self.vocab.eos_index]
                t2 = [self.vocab.sos_index] + t2 + [self.vocab.eos_index]
                t1_len = min(len(t1), self.seq_len)
                t2_len = min(len(t2), self.seq_len)
                padding_t1 = [self.vocab.pad_index for _ in range(self.seq_len - len(t1))]
                padding_t2 = [self.vocab.pad_index for _ in range(self.seq_len - len(t2))]
                t1.extend(padding_t1)
                t2.extend(padding_t2)
                t1, t2 = t1[:self.seq_len], t2[:self.seq_len]
                # t1=np.array(t1).tostring()
                # t2 = np.array(t2).tostring()
                # t1_len = np.array(t1_len).tostring()
                # t2_len = np.array(t2_len).tostring()
                # label = np.array(label).tostring()
                # t1 = np.array(t1)
                # t2 = np.array(t2)
                # t1_len = np.array(t1_len)
                # t2_len = np.array(t2_len)
                # label = np.array(label)
                features = collections.OrderedDict()
                features["sent1_word"] = self._int64_feature(t1)
                features["sent2_word"] = self._int64_feature(t2)
                features["sent1_len"] = self._int64_feature([t1_len])
                features["sent2_len"] = self._int64_feature([t2_len])
                features["label"] = self._int64_feature([label])

                example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(example.SerializeToString())
                index+=1
                if index%100000==0:
                    print(index)
                    print(t1,'\t\t',t2,'\t\t',t1_len,'\t\t',t2_len,'\t\t',label)
        writer.close()
        writer_index.write(str(index))
        writer_index.close()



    def _int64_feature(self,value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))
    def _int64_feature1(self,value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _bytes_feature(self,value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


