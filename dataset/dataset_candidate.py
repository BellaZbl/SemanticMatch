from torch.utils.data import Dataset
import tqdm
import torch
import random
import numpy as np
from collections import defaultdict


class DatasetCandidate(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, candidate_dict,label_dict,candidata_list,infer_list,encoding="utf-8" ,corpus_lines=None,label_vocab=None,
                 infer_id_sent=None, candidate_id_sent=None):
        self.vocab = vocab
        self.seq_len = seq_len
        self.candidate=candidate_dict
        self.candidate_num=1
        self.label_dict=label_dict
        self.corpus_path=corpus_path
        self.infer_list=infer_list
        self.candidata_list=candidata_list
        # if infer_id_sent and candidate_id_sent:
        self.infer_id_sent=infer_id_sent
        self.candidate_id_sent=candidate_id_sent
        self.get_datas()

    def get_datas(self):
        if self.infer_list and self.candidata_list:
            datas_new=[]
            for item in range(len(self.infer_list)):
                t1 = self.infer_list[item]
                if self.infer_id_sent and t1 in self.infer_id_sent:t1_id=self.infer_id_sent[t1]
                else:t1_id= -1
                t1 = self.vocab.to_seq(t1)
                t1 = [self.vocab.sos_index] + t1 + [self.vocab.eos_index]
                padding_t1 = [self.vocab.pad_index for _ in range(self.seq_len - len(t1))]
                t1_len = min(len(t1), self.seq_len)
                t1.extend(padding_t1)
                t1 = t1[:self.seq_len]
                for ele in self.candidata_list:
                    if self.candidate_id_sent and ele in self.candidate_id_sent: t2_id = self.candidate_id_sent[ele]
                    else:t2_id = -1
                    t2 = self.vocab.to_seq(ele)
                    t2 = [self.vocab.sos_index] + t2 + [self.vocab.eos_index]
                    t2_len = min(len(t2), self.seq_len)
                    padding_t2 = [self.vocab.pad_index for _ in range(self.seq_len - len(t2))]
                    t2.extend(padding_t2)
                    t2 = t2[:self.seq_len]

                    datas_new.append([t1,t2,t1_len,t2_len,t1_id,t2_id])
            self.datas=datas_new

        else:
            with open(self.corpus_path, "r", encoding='utf-8') as f:
                self.datas = [line.lower().replace('\n','').split("\t")
                              for line in tqdm.tqdm(f, desc="Loading Dataset", total=None) if line]
            datas_new=[]
            for item in range(len(self.datas)):
                t1, _, true_label = self.datas[item][:]
                t1 = self.vocab.to_seq(t1)
                t1 = [self.vocab.sos_index] + t1 + [self.vocab.eos_index]
                padding_t1 = [self.vocab.pad_index for _ in range(self.seq_len - len(t1))]
                t1_len = min(len(t1), self.seq_len)
                t1.extend(padding_t1)
                t1= t1[:self.seq_len]
                print(t1)
                for k, v in self.candidate.items():
                    v = list(v)
                    random.shuffle(v)
                    candidate_num = min(self.candidate_num, len(v))
                    vs = v[:candidate_num]
                    for t2 in vs:
                        t2 = self.vocab.to_seq(t2)
                        t2 = [self.vocab.sos_index] + t2 + [self.vocab.eos_index]
                        t2_len = min(len(t2), self.seq_len)
                        padding_t2 = [self.vocab.pad_index for _ in range(self.seq_len - len(t2))]
                        t2.extend(padding_t2)
                        t2 = t2[:self.seq_len]
                        datas_new.append([t1,t2,t1_len,t2_len,self.label_dict[true_label],self.label_dict[k]])
            self.datas=datas_new

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        if self.infer_list and self.candidata_list:
            t1, t2, t1_len, t2_len,t1_id,t2_id = self.datas[item][:]
            output = {"sent1_word": t1,
                      "sent2_word": t2,
                      "sent1_len": t1_len,
                      "sent2_len": t2_len,
                      "sent1_id": t1_id,
                      "sent2_id": t2_id
                      }

            return {key: torch.tensor(value) for key, value in output.items()}
        else:
            t1, t2, t1_len, t2_len,true_label,candidate_label=self.datas[item][:]
            output = {"sent1_word":t1,
                      "sent2_word":t2,
                      "sent1_len":t1_len,
                      "sent2_len": t2_len,
                      "true_label": true_label,
                      "candidate_label":candidate_label,
                       }

            return {key: torch.tensor(value) for key, value in output.items()}



class DatasetPairs(Dataset):
    def __init__(self, vocab, seq_len, label_dict,data_pairs_list):
        self.vocab = vocab
        self.seq_len = seq_len
        self.data_pairs_list=data_pairs_list
        self.candidate_num=1
        self.label_dict=label_dict
        self.get_datas()

    def get_datas(self):

        datas_new=[]
        for item in range(len(self.data_pairs_list)):
            t1,t2 = self.data_pairs_list[item][:]

            t1 = self.vocab.to_seq(t1)
            t1 = [self.vocab.sos_index] + t1 + [self.vocab.eos_index]
            padding_t1 = [self.vocab.pad_index for _ in range(self.seq_len - len(t1))]
            t1_len = min(len(t1), self.seq_len)
            t1.extend(padding_t1)
            t1 = t1[:self.seq_len]

            t2 = self.vocab.to_seq(t2)
            t2 = [self.vocab.sos_index] + t2 + [self.vocab.eos_index]
            t2_len = min(len(t2), self.seq_len)
            padding_t2 = [self.vocab.pad_index for _ in range(self.seq_len - len(t2))]
            t2.extend(padding_t2)
            t2 = t2[:self.seq_len]

            datas_new.append([t1,t2,t1_len,t2_len])
        self.datas=datas_new


    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        t1, t2, t1_len, t2_len = self.datas[item][:]
        output = {"sent1_word": t1,
                  "sent2_word": t2,
                  "sent1_len": t1_len,
                  "sent2_len": t2_len,
                  }

        return {key: torch.tensor(value) for key, value in output.items()}

