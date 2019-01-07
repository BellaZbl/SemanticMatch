word2id={}
id2word={}
for ele in open('./word_dict.txt','r',encoding='utf-8').readlines():
    ele=ele.replace('\n','')
    eles=ele.split(' ')
    word2id[eles[0]]=eles[1]
    id2word[eles[1]]=eles[0]

fw_train=open('qqa_qa_train.txt','w',encoding='utf-8')
fw_test=open('qqa_qa_test.txt','w',encoding='utf-8')
fw_valid=open('qqa_qa_valid.txt','w',encoding='utf-8')

id2sent={}
for ele in open('corpus_preprocessed.txt','r',encoding='utf-8').readlines():
    ele=ele.replace('\n','')
    try:
        id_,sent=ele.split('\t')[:]
        sent_ch=' '.join(id2word[e] for e in str(sent).split(' '))
        id2sent[id_]=sent_ch
    except:
        print(ele)


for ele in open('./relation_train.txt','r',encoding='utf-8').readlines():
    ele=ele.replace('\n','')
    label,id1,id2=ele.split(' ')[:]
    try:
        fw_train.write(id2sent[id1]+'\t\t'+id2sent[id2]+'\t\t'+label+'\n')
    except:
        print(ele)

for ele in open('./relation_test.txt','r',encoding='utf-8').readlines():
    ele=ele.replace('\n','')
    label,id1,id2=ele.split(' ')[:]
    fw_test.write(id2sent[id1]+'\t\t'+id2sent[id2]+'\t\t'+label+'\n')

for ele in open('./relation_valid.txt','r',encoding='utf-8').readlines():
    ele=ele.replace('\n','')
    label,id1,id2=ele.split(' ')[:]
    fw_valid.write(id2sent[id1]+'\t\t'+id2sent[id2]+'\t\t'+label+'\n')