from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
import numpy as np

from tensorflow.contrib import rnn

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

training_iters = 150
batch_size = 25
display_step = 1
classes = 3
n_hidden = 10
Pool_data, Pool_labels = get_data(data_path,labels_path)

Pool_data = reorganize(Pool_data,100)
test_data = Pool_data[500:]
Pool_data = Pool_data[:500]
n_steps = Pool_data.shape[1]
n_input = Pool_data.shape[2]

print(str(n_steps) + " steps " + str(n_input) + " input ")
weights = [tf.Variable(tf.random_normal([n_steps,2 * n_hidden, n_input]))]

biases = [tf.Variable(tf.random_normal([n_steps, n_input]))]

n_steps = Pool_data.shape[1]
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
        for i in range(1, len(Pool_data), batch_size):
            batch_x = Pool_data[i:i + cur_batch_size]
            batch_y = Pool_data[i:i + cur_batch_size].reshape(n_steps,cur_batch_size,n_input)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            cur_batch_size = Pool_data[i+batch_size:i + 2*batch_size].shape[0]
        # Reshape data to get 28 seq of 28 elements
        # batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        if epoch % display_step == 0:
            # Calculate batch accuracy
            # SNP_data_train.reshape(n_steps,495,n_input)
            acc = sess.run(cost, feed_dict={x: Pool_data, y: Pool_data.reshape(n_steps,Pool_data.shape[0],n_input)})
            test_acc = sess.run(cost, feed_dict={x: test_data, y: test_data.reshape(n_steps,test_data.shape[0],n_input)})

            print(" Iter: " + str(epoch) + ", Training Cost= " + \
                  "{:.5f}".format(acc/2) + ", Testing Cost: " + str(test_acc/2) + ' /n')


print("Optimization Finished!")
