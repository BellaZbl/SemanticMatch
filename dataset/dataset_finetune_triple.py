from torch.utils.data import Dataset
import tqdm
import torch
import random
import numpy as np


class BERTDatasetFineTuneTriple(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None,label_vocab=None):
        self.vocab = vocab
        self.seq_len = seq_len
        self.label_vocab=label_vocab
        with open(corpus_path, "r", encoding=encoding) as f:
            self.datas = [line.lower().replace('\n','').split("\t\t")
                          for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines) if line]
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        try:
            t0,tpos,tneg=self.datas[item][:]

            # [CLS] tag = SOS tag, [SEP] tag = EOS tag
            t0=self.vocab.to_seq(t0)
            tpos=self.vocab.to_seq(tpos)
            tneg=self.vocab.to_seq(tneg)

            t0 = [self.vocab.sos_index] + t0 + [self.vocab.eos_index]
            tpos = [self.vocab.sos_index] + tpos + [self.vocab.eos_index]
            tneg = [self.vocab.sos_index] + tneg + [self.vocab.eos_index]

            t0_len=min(len(t0),self.seq_len)
            tpos_len=min(len(tpos),self.seq_len)
            tneg_len=min(len(tneg),self.seq_len)

            padding_t0 = [self.vocab.pad_index for _ in range(self.seq_len - len(t0))]
            padding_tpos = [self.vocab.pad_index for _ in range(self.seq_len - len(tpos))]
            padding_tneg = [self.vocab.pad_index for _ in range(self.seq_len - len(tneg))]

            t0.extend(padding_t0)
            tpos.extend(padding_tpos)
            tneg.extend(padding_tneg)

            t0,tpos,tneg=t0[:self.seq_len],tpos[:self.seq_len],tneg[:self.seq_len]

            output = {"sent_0_word":t0,
                      "sent_pos_word":tpos,
                      "sent_neg_word": tneg,
                      "sent_0_len":t0_len,
                      "sent_pos_len": tpos_len,
                      "sent_neg_len": tneg_len,
                       }

            return {key: torch.tensor(value) for key, value in output.items()}
        except:
            print(self.datas[item])

