import numpy as np
# # import tensorflow as tf
# # s=np.array([2,3,1,0,0,0])
# #
# # print(s.shape)
# # ss=tf.get_variable(initializer=tf.constant_initializer(s,dtype=tf.int32),shape=(6,),dtype=tf.int32,name='12')
# #
# # s1=tf.cast(ss,tf.bool)
# #
# # with tf.Session() as sess:
# #     sess.run(tf.global_variables_initializer())
# #     print(sess.run(s1))
# print(20/3)
#
# import pickle as pkl
#
# sent_dict=pkl.load(open('sent_dict.p','rb'))
#
# for k,v in sent_dict.items():
#     v.sort(key=lambda x:x[1],reverse=True)
#     print(k,'\t\t',v,'\n')
import tensorflow as tf
import os

def load_file(example_list_file):
    lines = np.genfromtxt(example_list_file,delimiter="\t\t",dtype=str,encoding='utf-8')
    print(lines)
    sent1s,sent2s,labels = [],[],[]
    for sent1,sent2,label in lines:
        sent1s.append(sent1)
        sent2s.append(sent2)
        labels.append(label)
    #convert to numpy array
    return np.asarray(examples),np.asarray(labels),len(lines)

def _int64_feature(value):
     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def trans2tfRecord(trainFile,name,output_dir):
    if not os.path.exists(output_dir) or os.path.isfile(output_dir):
        os.makedirs(output_dir)
    _examples,_labels,examples_num = load_file(trainFile)
    filename = name + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(filename)
    for i,[example,label] in enumerate(zip(_examples,_labels)):
        print("NO{}".format(i))
        #need to convert the example(bytes) to utf-8
        example = example.decode("UTF-8")
        example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw':_bytes_feature(image_raw),
                'height':_int64_feature(image.shape[0]),
                 'width': _int64_feature(32),
                'depth': _int64_feature(32),
                 'label': _int64_feature(label)
                }))
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    trans2tfRecord(trainFile='./data/sim_test_char_small.txt',name='train',output_dir='./oo')