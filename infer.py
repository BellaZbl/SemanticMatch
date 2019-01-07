from model.BimpmModel import Bimpm
from model.MatchPyramid import MatchPyramid
from model.LstmMatchPyramid import LstmMatchPyramid
import sys
sys.path.append('.')
from model.SiameseCNN import SiameseCNN
from model.SiameseLstm import SiameseLstm
from model.Transformer import TransformerModel
from model.LstmMatchPyramid import LstmMatchPyramid
from dataset import BERTDataset,WordVocab,BERTDatasetFineTune,DatasetCandidate
import tensorflow as tf
import pickle as pkl
import argparse
from torch.utils.data import DataLoader
import os
from collections import defaultdict
import jieba
from data.utils import tools
from pprint import pprint
PATH=os.path.split(os.path.realpath(__file__))[0]
jieba.load_userdict(PATH+"/data/user_dict.txt")
from collections import Counter

import logging
logger=logging.getLogger()
logger.setLevel(logging.INFO)
logfile=PATH+'/log/transformer_log.txt'
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

def get_candidates(candidate_path):
    candidate_dict=defaultdict(set)
    for line in open(candidate_path,'r',encoding='utf-8').readlines():
        line=line.replace('\n','')
        sent=line.split('\t')[0]
        label=line.split('\t')[-1]
        candidate_dict[label].add(sent)
    return candidate_dict

def get_label_dict(candidate_path,test_path):
    label_dict={}
    index=0
    for ele in open(candidate_path, 'r', encoding='utf-8'):
        label = ele.replace('\n', '').split('\t')[-1]
        if label not in label_dict:
            label_dict[label] = index
            index += 1
    for ele in open(test_path, 'r', encoding='utf-8'):
        label = ele.replace('\n', '').split('\t')[-1]
        if label not in label_dict:
            label_dict[label] = index
            index += 1
    return label_dict

def train():

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.vocab_path = PATH+'/data/vocab.big'
    args.candidate_dataset = PATH+'/data/finetune_train_new1.big'
    args.test_dataset = PATH+'/data/finetune_test_new2.txt'
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

    print("load candidate_dict",args.candidate_dataset)
    # candidate_dict=get_candidates(args.candidate_dataset)

    candidata_list=pkl.load(open(PATH+"/infer/candidate_list.p",'rb'))
    infer_list=pkl.load(open(PATH+"/infer/test_list.p",'rb'))

    infer_id_sent=pkl.load(open(PATH+"/infer/test_id_sent.p",'rb'))
    candidate_id_sent=pkl.load(open(PATH+"/infer/train_id_sent.p",'rb'))

    _logger.info("infer_size:%s candidate_size:%s"%(len(infer_list),len(candidata_list)))
    print("loading label dict")
    label_dict=pkl.load(open(PATH+"/label_vocab.p",'rb'))
    args.id2label={v:k for k,v in label_dict.items()}
    print("Loading Vocab", args.vocab_path)
    vocab = WordVocab.load_vocab(args.vocab_path)
    id2word={v:k for k,v in vocab.stoi.items()}
    args.id2word=id2word
    print("Vocab Size: ", len(vocab))
    args.vocab_size=len(vocab)

    print("Loading test Dataset", args.test_dataset)
    infer_data = DatasetCandidate(args.test_dataset, vocab, seq_len=args.seq_len, candidate_dict=None,label_dict=label_dict,candidata_list=candidata_list,
                                  infer_list=infer_list,infer_id_sent=infer_id_sent,candidate_id_sent=candidate_id_sent)
    print(infer_data.__len__())
    print("Creating Dataloader")
    infer_data_loader = DataLoader(infer_data, batch_size=args.batch_size, num_workers=args.num_workers)
    print(len(infer_data_loader))


    args.num_classes=2
    # for e in infer_data_loader:
    #     print(e['true_label'][0],e['candidate_label'][0])
    args.model_name="transformer"

    if args.model_name == 'bimpm':
        model=Bimpm(args,'bimpm')
    elif args.model_name == 'mp':
        model=MatchPyramid(args,'matchpyramid')
    elif args.model_name == 'lstm_mp':
        model = LstmMatchPyramid(args, "lstm_matchpyramid")
    elif args.model_name == 'siamese_cnn':
        model = SiameseCNN(args,"SiameseCNN")
    elif args.model_name == 'siamese_lstm':
        model = SiameseLstm(args, "SiameseLstm")
    elif args.model_name == 'transformer':
        model = TransformerModel(args, "Transformer")


    model.build_placeholder()
    model.build_model()
    model.build_accuracy()
    sent_dict=model.infer(infer_data_loader, restore_model_path=PATH + "/output/%s_small.ckpt"%args.model_name)
    pkl.dump(sent_dict,open(PATH+'/sent_dict.p','wb'))

    id2infer={v:k for k,v in infer_id_sent.items()}
    id2candidate={v:k for k,v in candidate_id_sent.items()}

    infer_intent=pkl.load(open(PATH+"/infer/test_intent_new.p","rb"))
    candidate_intent=dict(pkl.load(open(PATH+"/infer/candidate_dict_intent.p",'rb')))
    mod='candidate_single'
    if mod=='candidate_list':
        all_index,index=0,0
        for k,v in sent_dict.items():
            if k in id2infer:
                infer_sent=id2infer[k]
                if infer_sent.lower() in infer_intent:
                    infer_intent1=infer_intent[infer_sent.lower()]
                    all_index += 1
                    intents=[]
                    for ele in v:
                        candidate_sent=id2candidate[int(ele[0])]
                        # print(ele[0],'\t\t',candidate_sent)
                        if ele[1]>=0.95:
                            if candidate_sent.lower() in candidate_intent:
                                candidate_intent1=candidate_intent[str(candidate_sent).lower()]
                                intents.append(candidate_intent1)
                            else:
                                print('no sent:%s'%candidate_sent.lower())
                    ss=Counter(intents)
                    ss=[[k,v] for k,v in ss.items()]
                    ss.sort(key=lambda x: x[1], reverse=True)

                    rg=[e[0] for e in ss[:2]]
                    if infer_intent1 in rg:
                        index+=1
                    else:
                        print(infer_sent,'\t\t',infer_intent1,'\t\t',ss,'\n')
        print('%s/%s' % (index, all_index))
    elif mod=='candidate_single':
        all_index, index = 0, 0
        for k, v in sent_dict.items():
            if k in id2infer:
                infer_sent = id2infer[k]
                if infer_sent.lower() in infer_intent:
                    infer_intent1 = infer_intent[infer_sent.lower()]
                    all_index += 1
                    intents = []
                    dd=[]
                    for ele in v[:2]:
                        candidate_sent = id2candidate[int(ele[0])]
                        dd.append(candidate_sent)
                        # print(ele[0],'\t\t',candidate_sent)
                        if candidate_sent.lower() in candidate_intent:
                            candidate_intent1 = candidate_intent[str(candidate_sent).lower()]
                            intents.append(candidate_intent1)
                        else:
                            print('no sent:%s' % candidate_sent.lower())
                    ss = Counter(intents)
                    ss = [[k, v] for k, v in ss.items()]
                    ss.sort(key=lambda x: x[1], reverse=True)

                    rg = [e[0] for e in ss[:2]]
                    if infer_intent1 in rg:
                        index += 1
                    else:
                        dd="\t\t".join(dd)
                        print(infer_sent, '\t\t', infer_intent1, '\t\t', ss, '\t\t',dd,'\n')

        print('%s/%s' % (index, all_index))


if __name__ == '__main__':
    train()