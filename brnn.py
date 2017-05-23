from __future__ import print_function

import numpy as np

import tensorflow as tf


def get_data(datafile, labelfile,seed=21,t=500,delim=',',ty=np.float32):
    print('entered get data')
    data   = np.loadtxt(datafile,delimiter=delim,dtype=np.float32)
    print('loaded data')
    target = np.loadtxt(labelfile,delimiter=delim,dtype=np.int32)
    print('loaded label')
    # Prepend the column of 1s for bias
    N, M  = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data
    # Convert into one-hot vectors
    num_labels = len(np.unique(target))
    #all_Y = np.eye(num_labels)[target]  # One liner trick!
    np.random.seed(seed)
    p = np.random.permutation(N)
    data = data[p]
    target = target[p]
    train_d = data[:t]
    test_d = data[t:]
    #targets = make_one_hot(target,num_labels)
    targets = target
    train_t = targets[:t]
    test_t = targets[t:]
    print('returning data')
    return data,target

'''
A Bidirectional Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''


import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

'''
To classify images using a bidirectional recurrent neural network, we consider
every image row as a sequence of pixels. Because MNIST image shape is 28*28px,
we will then handle 28 sequences of 28 steps for every sample.
'''

# Parameters
learning_rate = 0.001
training_iters = 10
batch_size = 20
display_step = 10

# Network Parameters
n_input = 107 # MNIST data input (img shape: 28*28)
n_steps = 70 # timesteps
n_hidden = 300 # hidden layer num of features
n_classes = 2 # MNIST total classes (0-9 digits)

# tf Graph input


# Define weights



def BiRNN(x, weights, biases):

    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']
def make_one_hot(target,labels):
    targets = np.zeros((len(target),labels))
    targets[np.arange(len(target)),target-1] = 1
    return targets

def organize_data(datas,step_size):
    print('hi')
    r = []
    for d in datas:
        subjects = d.shape[0]
        print(d.shape)
        data_size = d.shape[1] - 2
        timesteps = data_size / step_size
        r.append(d[:,:data_size].reshape(subjects,timesteps,step_size))
    return r
Pool_data,Pool_labels = get_data('/home/lhd/PycharmProjects/tf_multimodal/data_right.txt','/home/lhd/PycharmProjects/tf_multimodal/labels_right.txt',delim=',',ty=np.int32)

idxs = [[range(0, 495), range(495, 550), range(550, 583)],
        [range(0, 550), range(0, 55), range(550, 583)],
        [range(0, 55) + range(110, 550), range(55, 110), range(550, 583)],
        [range(0, 110) + range(165, 550), range(110, 165), range(550, 583)],
        [range(0, 165) + range(220, 550), range(165, 220), range(550, 583)],
        [range(0, 220) + range(275, 550), range(220, 275), range(550, 583)],
        [range(0, 275) + range(330, 550), range(275, 330), range(550, 583)],
        [range(0, 330) + range(385, 550), range(330, 385), range(550, 583)],
        [range(0, 385) + range(440, 550), range(385, 440), range(550, 583)],
        [range(0, 440) + range(495, 550), range(440, 495), range(550, 583)]
        ]
for i in range(len(idxs)):
    weights = {
        # Hidden layer weights => 2*n_hidden because of forward + backward cells
        'out': tf.Variable(tf.random_normal([2 * n_hidden, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    x = tf.placeholder("float", [None, n_steps, n_input])
    y = tf.placeholder("float", [None, n_classes])
    SNP_data_train,SNP_data_test = organize_data([Pool_data[idxs[i][0]],Pool_data[idxs[i][1]]],107)
    pred = BiRNN(x, weights, biases)
    print("break")
    print(SNP_data_test.shape)
    print(SNP_data_train.shape)

    SNP_label_train = make_one_hot(Pool_labels[idxs[i][0]], 2)
    SNP_label_test = make_one_hot(Pool_labels[idxs[i][1]], 2)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations

        for epoch in range(training_iters):
            for i in range(1,len(SNP_data_train),batch_size):
                #print(i)
                batch_x, batch_y = SNP_data_train[i:i+batch_size],SNP_label_train[i:i+batch_size]
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            # Reshape data to get 28 seq of 28 elements
            #batch_x = batch_x.reshape((batch_size, n_steps, n_input))
            # Run optimization op (backprop)
            if epoch % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: SNP_data_train, y: SNP_label_train})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                print("Iter " + str(epoch*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
                print("Testing Accuracy:", \
                      sess.run(accuracy, feed_dict={x: SNP_data_test, y: SNP_label_test}))

        print("Optimization Finished!")

        # Calculate accuracy for 128 mnist test images
        test_len = SNP_data_test.shape[0]
        test_data = SNP_data_test#mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
        test_label = SNP_label_test#mnist.test.labels[:test_len]
        print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
    tf.reset_default_graph()
