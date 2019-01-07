from torch.utils.data import Dataset
import tqdm
import torch
import random
import numpy as np


class BERTDatasetInfer(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None,label_vocab=None):
        self.vocab = vocab
        self.seq_len = seq_len
        self.label_vocab=label_vocab
        with open(corpus_path, "r", encoding=encoding) as f:
            self.datas = [line.lower().replace('\n','').split("\t")
                          for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines) if line]
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        t1,t2,label_=self.datas[item][:]
        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1=self.vocab.to_seq(t1)
        t2=self.vocab.to_seq(t2)
        t1 = [self.vocab.sos_index] + t1 + [self.vocab.eos_index]
        t2 = [self.vocab.sos_index]+t2 + [self.vocab.eos_index]
        t1_len=min(len(t1),self.seq_len)
        t2_len=min(len(t2),self.seq_len)
        padding_t1 = [self.vocab.pad_index for _ in range(self.seq_len - len(t1))]
        padding_t2 = [self.vocab.pad_index for _ in range(self.seq_len - len(t2))]
        t1.extend(padding_t1)
        t2.extend(padding_t2)
        t1,t2=t1[:self.seq_len],t2[:self.seq_len]

        output = {"sent1_word":t1,
                  "sent2_word":t2,
                  "sent1_len":t1_len,
                  "sent2_len": t2_len,
                  "label": self.label_vocab[label_],
                   }

        return {key: torch.tensor(value) for key, value in output.items()}

