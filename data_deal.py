from collections import defaultdict
data_dict=defaultdict(list)
for ele in open('./data/finetune_train.big','r',encoding='utf-8').readlines():
    line=ele.replace('\n','')
    lines=line.split('\t')
    sent,label=line[0],line[-1]
    data_dict[label].append(sent)