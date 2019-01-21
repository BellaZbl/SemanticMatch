import csv
import random
import jieba
csvfiles=['atec_nlp_sim_train_add.csv','atec_nlp_sim_train.csv']

fw_train=open('atec_train.txt','w',encoding='utf-8')
fw_dev=open('atec_dev.txt','w',encoding='utf-8')

data=[]
for csvfile in csvfiles:
    csv_reader = csv.reader(open(csvfile,'r',encoding='utf-8'))
    for ele in csv_reader:
        try:
            eles=ele[0].replace('\n','').split('\t')
            sent1,sent2,label=eles[1],eles[2],eles[3]

            data.append([sent1,sent2,label])
        except :
            ele_=''.join(ele)
            try:
                eles = ele_.replace('\n', '').split('\t')
                sent1, sent2, label = eles[1], eles[2], eles[3]
                data.append([sent1, sent2, label])
            except:
                print([ele_])

random.shuffle(data)

split_rate=0.2

num=int(len(data)*split_rate)
s=''
for ele in data[:num]:
    sent1=ele[0]
    sent2=ele[1]
    label=ele[2]

    sent1_=' '.join([e for e in jieba.cut(sent1)])
    sent2_=' '.join([e for e in jieba.cut(sent2)])

    fw_dev.write(sent1_+'\t\t'+sent2_+'\t\t'+label+'\n')

for ele in data[num::]:
    sent1 = ele[0]
    sent2 = ele[1]
    label = ele[2]

    sent1_ = ' '.join([e for e in jieba.cut(sent1)])
    sent2_ = ' '.join([e for e in jieba.cut(sent2)])

    fw_train.write(sent1_ + '\t\t' + sent2_ + '\t\t' + label + '\n')