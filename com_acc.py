import pickle as pkl
from collections import Counter
test_dict=pkl.load(open('test_dict.p','rb'))
sent_dict=pkl.load(open('sent_dict.p','rb'))
one_all,one_right,two_all,two_right,three_all,three_right=0,0,0,0,0,0

for sent_key,v_list in sent_dict.items():
    sent_key = str(sent_key).replace(" ", "").replace("<sos>", "").replace("<eos>","")
    cc=Counter(v_list)
    print(sent_key,'\t\t',cc,'\n')
    intent = '000'
    for k, v in cc.items():
        if v >= 5:
            intent = k
    if sent_key in test_dict:
        intent_test,label_test=test_dict[sent_key]
        print(intent_test,"\t\t",intent,'\n')
        if label_test == '1':
            one_all+=1
            if intent_test==intent:
                one_right+=1

        if label_test == '2':
            two_all += 1
            if intent_test == intent:
                two_right += 1

        if label_test == '3':
            three_all += 1
            if intent_test == intent:
                three_right += 1

    print("one:%s  two:%s three:%s" % (str(one_right) + "/" + str(one_all), str(two_right) + "/" + str(two_all), str(three_right) + "/" + str(three_all)))
