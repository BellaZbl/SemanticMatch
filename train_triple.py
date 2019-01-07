from model.BimpmModel import Bimpm
from model.MatchPyramid import MatchPyramid
from model.LstmMatchPyramid import LstmMatchPyramid
from model.SiameseCNN import SiameseCNN
from model.SiameseCNNTriple import SiameseCNNTriple
from model.SiameseLstm import SiameseLstm
from model.LstmMatchPyramid import LstmMatchPyramid
from dataset import BERTDataset,WordVocab,BERTDatasetFineTune,BERTDatasetFineTuneTriple
import tensorflow as tf
import pickle as pkl
import argparse
from torch.utils.data import DataLoader
import os
from pprint import pprint
PATH=os.path.split(os.path.realpath(__file__))[0]

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

def train():

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.vocab_path = PATH+'/data/vocab.big'
    args.label_vocab_path = PATH+'/data/label_vocab.big'
    args.train_dataset = PATH+'/data/triple_train_11_9_char_small.txt'
    args.test_dataset = PATH+'/data/triple_test_11_9_char.txt'
    args.output_path = PATH+'/output'
    args.seq_len = 50
    args.corpus_lines = None
    args.batch_size = 256
    args.num_workers = 6
    args.emb_size=100
    args.highway_layer_num=1
    args.hidden = 256
    args.layers = 4
    args.attn_heads = 4
    args.lr = 0.001
    args.adam_beta1 = 0.9
    args.adam_beta2 = 0.999
    args.adam_weight_decay = 0.01
    args.with_cuda = True
    args.log_freq = 10
    args.epochs = 100
    args.sinusoid=False
    args.loss='softmax_loss'
    args.opt='adam'


    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)


    print("build vocab")
    build_vocab(args.train_dataset,args.vocab_path)

    print("Loading Vocab", args.vocab_path)
    vocab = WordVocab.load_vocab(args.vocab_path)
    args.id2vocab={v:k for k,v in vocab.stoi.items()}
    print("Vocab Size: ", len(vocab))
    args.vocab_size=len(vocab)

    print("Loading Train Dataset", args.train_dataset)
    train_dataset = BERTDatasetFineTuneTriple(args.train_dataset, vocab, seq_len=args.seq_len, corpus_lines=args.corpus_lines)

    # print("Loading Test Dataset", args.test_dataset)
    test_dataset = BERTDatasetFineTuneTriple(args.test_dataset, vocab,
                               seq_len=args.seq_len) if args.test_dataset is not None else None
    #
    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    print("train_data_loader length",len(train_data_loader))

    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) \
        if test_dataset is not None else None

    args.num_classes=2
    model_name='siamese_cnn_triple'
    if model_name == 'bimpm':
        model=Bimpm(args,'bimpm')
    elif model_name == 'mp':
        model=MatchPyramid(args,'matchpyramid')
    elif model_name == 'lstm_mp':
        model = LstmMatchPyramid(args, "lstm_matchpyramid")
    elif model_name == 'siamese_cnn_triple':
        model = SiameseCNNTriple(args,"SiameseCNN")
    elif model_name == 'siamese_lstm':
        model = SiameseLstm(args,"SiameseLstm")


    model.build_placeholder()
    model.build_model()
    # model.build_accuracy()
    model.build_loss()
    model.build_op()
    # model.get_sent_emb(train_data_loader,restore_model_path=PATH+"/output/siamese_cnn_triple_cnn.ckpt",sent_emb_name="sent_emb_siamese.p")
    model.train(train_data_loader, test_data_loader,restore_model=None, save_model=PATH + "/output/%s_lstm.ckpt"%model_name)


if __name__ == '__main__':
    train()