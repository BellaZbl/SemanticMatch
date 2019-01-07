import tensorflow as tf
import numpy as np
from tensorflow.contrib import seq2seq


def load_data(fname):
    with open(fname, 'r') as f:
        text = f.read()

    data = text.split()
    return data

text = load_data('data/split.txt')

vocab = set(text)
vocab_to_int = {w: idx for idx, w in enumerate(vocab)}
int_to_vocab = {idx: w for idx, w in enumerate(vocab)}

int_text = [vocab_to_int[w] for w in text]

def get_inputs():
    '''
    构建输入层
    '''
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    return inputs, targets, learning_rate

def get_init_cell(batch_size,rnn_size):
    lstm=tf.contrib.rnn.BasicLSTMCell(rnn_size)
    cell=tf.contrib.rnn.MultiRNNCell([lstm])

    initial_state=cell.zero_state(batch_size,tf.float32)
    initial_state=tf.identity(initial_state,'initial_state')
    return cell,initial_state

def get_embed(input_data,vocab_size,embed_dim):
    embedding=tf.Variable(tf.random_uniform([vocab_size,embed_dim],-1,1))
    embed=tf.nn.embedding_lookup(embedding,input_data)

    return embed

def build_rnn(cell,inputs):
    outputs,final_state=tf.nn.dynamic_rnn(cell,inputs,dtype=tf.float32)

    final_state=tf.identity(final_state,'final_state')
    return outputs,final_state



def build_nn(cell,rnn_size,input_data,vocab_size,embed_dim):
    embed=get_embed(input_data,vocab_size,embed_dim)
    outputs,final_state=build_rnn(cell,embed)

    logits=tf.contrib.layers.fully_connected(outputs,vocab_size,activation_fn=None)

    return logits,final_state

def get_batches(int_text,batch_size,seq_length):

    batch=batch_size*seq_length
    n_batch=len(int_text)//batch

    int_text=np.array(int_text[:batch*n_batch])

    int_text_targets=np.zeros_like(int_text)
    int_text_targets[:-1],int_text_targets[-1]=int_text[1:],int_text[0]

    x=np.split(int_text.reshape(batch_size,-1),n_batch,-1)
    y=np.split(int_text_targets.reshape(batch_size,-1),n_batch,-1)

    return np.stack((x,y),axis=1)

rnn_size=512
embed_dim=200
batch_size=64
seq_length=20
batches=get_batches(int_text,batch_size,seq_length)
save_dir='./save'
num_epochs=100
learning_rate=0.01
show_every_n_batches=50

train_graph=tf.Graph()
with train_graph.as_default():
    vocab_size=len(int_to_vocab)
    input_text,targets,lr=get_inputs()
    input_data_shape=tf.shape(int_text)

    cell,initial_state=get_init_cell(input_data_shape[0],rnn_size)
    logits,final_state=build_nn(cell,rnn_size,input_text,vocab_size,embed_dim)

    probs=tf.nn.softmax(logits,name='probs')

    cost=seq2seq.sequence_loss(logits,targets,tf.ones([input_data_shape[0],input_data_shape[1]]))

    optimizer=tf.train.AdamOptimizer(lr)

    gradients=optimizer.compute_gradients(cost)
    capped_gradients=[(tf.clip_by_value(grad,-1.,1.),var) for grad,var in gradients if grad is not None]
    train_op=optimizer.apply_gradients(capped_gradients)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(num_epochs):
        #  为什么循环完一次数据后要重新定义initial_state呢？
        state=sess.run(initial_state,{input_text:batches[0][0]})

        for batch_i,(x,y) in enumerate(batches):
            feed={
                input_text:x,
                targets:y,
                initial_state:state,
                lr:learning_rate}
            train_loss,state,_=sess.run([cost,final_state,train_op],feed)

        if (epoch*len(batches)+batch_i)%show_every_n_batches==0:
            print('Epoch {:>3} Batch {:>4}/{} train_loss={:.3f}'.format(
                epoch,
                batch_i,
                len(batches),
                train_loss))

    saver=tf.train.Saver()
    saver.save(sess,save_dir)
    print('Model Trained and Saved')












