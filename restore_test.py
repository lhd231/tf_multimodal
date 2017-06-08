import tensorflow as tf
import numpy as np
from utilities import *



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

training_iters = 150
batch_size = 50
display_step = 10

n_hidden = 10
learning_rate = .01
tf.reset_default_graph()
folder = "/export/mialab/users/nlewis/tf_multimodal/pretrain_model_UPENN_" + str(n_hidden) + "_" + str(learning_rate)+"/"

name = folder

data_path = "./data_right.txt"

Pool_data, _ = get_data(data_path,delim= ',')

Pool_data = reorganize(Pool_data,100)

n_steps = 100
n_input = 74

W1 = tf.Variable(tf.random_normal([n_steps, 2 * n_hidden, n_input]),name='RNN_weights_'+str(n_steps))
#X_train, X_test = get_data('/home/lhd/upenn/genodata_imputed_upenn_selected.txt', delim=' ', t=6500)
input_size = 7492#X_train.shape[1]
init = tf.global_variables_initializer()


#saver = tf.train.Saver()
with tf.Session() as sess:

    saver = tf.train.import_meta_graph(name + "model.ckpt.meta")
    saver.restore(sess, name + "model.ckpt")
    #saver.restore(sess, tf.train.latest_checkpoint("/home/lhd/tensorflow/For_Tensorflow_Multimodal/model_"+str(h_layer)+"_"+str(h_middle_layer)+"_"+str(h_decoder_layer)+'/'))
    Ops = sess.graph.get_operations()

    batch_x = Pool_data[i:i + 20]
    batch_y = Pool_data[i:i + cur_batch_size].reshape(n_steps, 20, n_input)
    sess.run(optimizer, feed_dict={X: batch_x, y: batch_y})

    print "complete"