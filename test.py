from model.BimpmModel import Bimpm
from model.MatchPyramid import MatchPyramid
from model.LstmMatchPyramid import LstmMatchPyramid
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
    args.vocab_path = PATH+'/data/vocab_8kw.big'
    args.candidate_dataset = PATH+'/data/finetune_train_new1.big'
    args.test_dataset = PATH+'/data/finetune_test_new2.txt'
    args.output_path = PATH+'/output'
    args.seq_len = 50
    args.corpus_lines = None
    args.batch_size = 2
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

    tool=tools()
    candidata_list_origin=['天气不错','是晴天',"是雨天",'红包支付不了',"红包收不了","不是提现问题","是提现问题呀","我在网上买资料，结果被骗了++给了钱找不见人了","我的卡绑了两个微信" ,"我的银行卡绑了两个微信","现 在 本 人 的 零钱 账户 不 能 转账 了","除 了 这 个 微信号 其 他 微信号 绑 定 的 银行卡 全 部 注 销","微信 零钱 用 不 了 也 发不了 小 小 的 红包","为什么我的零钱突然不能用了","修改密码"]
    infer_list_origin=["为什么红包不能发了",'是提现的错误',"这是个骗子，聊天记录在下方",'没发现绑了两个微信呀',"今天大太阳"]
    candidata_list=[]
    infer_list=[]

    for sent in candidata_list_origin:
        sent=tool.pre_deal_sent(sent,type='char')
        candidata_list.append(sent)

    for sent in infer_list_origin:
        sent = tool.pre_deal_sent(sent, type='char')
        infer_list.append(sent)

    print("loading label dict")
    label_dict={"0":0,'1':1}
    args.id2label={v:k for k,v in label_dict.items()}
    print("Loading Vocab", args.vocab_path)
    vocab = WordVocab.load_vocab(args.vocab_path)
    id2word={v:k for k,v in vocab.stoi.items()}
    print(id2word)
    args.id2word=id2word
    print("Vocab Size: ", len(vocab))
    args.vocab_size=len(vocab)

    print("Loading test Dataset", args.test_dataset)
    infer_data = DatasetCandidate(args.test_dataset, vocab, seq_len=args.seq_len, candidate_dict=None,label_dict=label_dict,candidata_list=candidata_list,infer_list=infer_list)
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
    model.infer(infer_data_loader, restore_model_path=PATH + "/output/%s_2kw.ckpt"%args.model_name)


if __name__ == '__main__':
    train()