from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
import numpy as np

from tensorflow.contrib import rnn

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

def get_data(datafile, labelfile,seed=21,delim=','):
    print('entered get data')
    data   = np.loadtxt(datafile,delimiter=delim,dtype=np.float32)
    print('loaded data')
    target = np.loadtxt(labelfile,delimiter=delim,dtype=np.int32)
    print('loaded label')

    np.random.seed(seed)
    p = np.random.permutation(data.shape[0])
    data = data[p]
    target = target[p]
    return data,target

def reorganize(X,timesteps=1):
    sample_size = X.shape[0]
    total_features = X.shape[1]
    removal = (total_features%timesteps)
    fixed_features = (total_features - removal)
    step_size = fixed_features / timesteps
    return X[:,:fixed_features].reshape(sample_size,timesteps,step_size)

def BiRNN(x, weight, bias):

    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    sample_size = x.shape[0]
    timesteps = x.shape[1]
    input = x.shape[2]
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Define lstm cells with tensorflow
    # Forward direction cell
    print("in birnn " + str(n_hidden))
    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    print("made left and right")
    # Get lstm cell output
    try:
        outputs, SL, SR = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        dtype=tf.float32)

    return tf.matmul(outputs, weight), SL, SR

data_path = "./data_right.txt"
labels_path="./labels_right.txt"

training_iters = 500
batch_size = 25
display_step = 5

n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features


print(str(n_steps) + " steps " + str(n_input) + " input ")
weights = [tf.Variable(tf.random_normal([n_steps,2 * n_hidden, n_input]))]

biases = [tf.Variable(tf.random_normal([n_steps, n_input]))]

x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [n_steps, None, n_input])

pred, _, _ = BiRNN(x, weights[0], biases[0])

#pred2 = tf.matmul(pred1,weights[1])

cost = tf.reduce_mean(tf.square(y-pred))

init = tf.global_variables_initializer()

optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    cur_batch_size = batch_size
    for epoch in range(training_iters):
            batch_x, _ = mnist.train.next_batch(batch_size)
            batch_x = batch_x.reshape(batch_size,n_steps,n_input)
            batch_y = batch_x.reshape(n_steps,batch_size,n_input)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            if (epoch) % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
    test_length = 128
    test_data = mnist.test.images[:test_length].reshape(test_length, n_steps, n_input)
    test_data_out = test_data.reshape((n_steps, test_length, n_input))
    print("Testing Accuracy:", \
          sess.run(cost, feed_dict={x: test_data, y: test_data_out}))
print("Optimization Finished!")
