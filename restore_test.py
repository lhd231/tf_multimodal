import tensorflow as tf
import numpy as np


def get_data(datafile, seed=21,t=500,delim=','):
    print('entered get data')
    data   = np.loadtxt(datafile,delimiter=delim,dtype=np.float32)
    print('loaded data')
    print data.shape
    #target = np.loadtxt(labelfile,delimiter=delim,dtype=np.int32)
    #print('loaded label')
    #print target.shape
    # Prepend the column of 1s for bias
    N, M  = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data
    np.random.seed(seed)
    p = np.random.permutation(N)
    data = data[p]
    train_d = data[:t]
    test_d = data[t:]
    print('returning data')
    print test_d.shape

    return train_d,test_d#,train_t,test_t

def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    biases = tf.random_normal([shape[1]])
    return tf.Variable(weights), tf.Variable(biases)

def return_weights(input_size,weight_sizes):
    X = tf.placeholder("float", shape=[None, input_size])
    #y = tf.placeholder("float", shape=[None, output_size])
    weights = []
    biases = []
    in_layer_size = input_size
    for size in weight_sizes:
        w,b=init_weights((in_layer_size, size))
        weights.append(w)
        biases.append(b)
        in_layer_size=size
    print len(weights)
    return(X,weights,biases)

h_layer = 1000
h_decoder_layer = 1000
h_middle_layer = 300
tf.reset_default_graph()
name = "/home/lhd/tensorflow/For_Tensorflow_Multimodal/model_"+str(h_layer)+"_"+str(h_middle_layer)+"_"+str(h_decoder_layer)+"/model.ckpt.meta"
#X_train, X_test = get_data('/home/lhd/upenn/genodata_imputed_upenn_selected.txt', delim=' ', t=6500)
input_size = 7492#X_train.shape[1]
init = tf.global_variables_initializer()
weights = tf.random_normal((input_size, h_layer), stddev=0.1)
biases = tf.random_normal([h_layer])
weights2 = tf.random_normal((h_layer, h_middle_layer), stddev=0.1)
biases2 = tf.random_normal([h_middle_layer])
weights3 = tf.random_normal((h_middle_layer, h_decoder_layer), stddev=0.1)
biases3 = tf.random_normal([h_decoder_layer])
weights4 = tf.random_normal((h_decoder_layer, input_size))
biases4 = tf.random_normal([input_size])
W1 = tf.Variable(weights, name='W1')
B1 = tf.Variable(biases, name='B1')
W2 = tf.Variable(weights2, name='W2')
B2 = tf.Variable(biases2, name='B2')
W3 = tf.Variable(weights3, name='W3')
B3 = tf.Variable(biases3, name='B3')
W4 = tf.Variable(weights4, name='W4')
B4 = tf.Variable(biases4, name='B4')
X, weights, biases = return_weights(input_size, [h_layer, h_middle_layer, h_decoder_layer])

#saver = tf.train.Saver()
with tf.Session() as sess:
    saver = tf.train.import_meta_graph(name)
    saver.restore(sess,tf.train.latest_checkpoint("/home/lhd/tensorflow/For_Tensorflow_Multimodal/model_1000_300_1000/"))
    #saver = tf.train.import_meta_graph('/home/lhd/tensorflow/For_Tensorflow_Multimodal/model.meta')
    #saver.restore(sess, tf.train.latest_checkpoint("/home/lhd/tensorflow/For_Tensorflow_Multimodal/model_"+str(h_layer)+"_"+str(h_middle_layer)+"_"+str(h_decoder_layer)+'/'))
    Ops = sess.graph.get_operations()
    print sess.graph.get_all_collection_keys()
    print Ops[1].name
    sess.run(tf.global_variables_initializer())
    graph = tf.get_default_graph()
    W1 = graph.get_tensor_by_name("W1:0")
    print W1.eval()
    print "complete"