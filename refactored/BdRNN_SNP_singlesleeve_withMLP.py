import tensorflow as tf
import numpy as np
from utilities import *
from tensorflow.contrib import rnn
import MLP
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#For the scripted example, we use the MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)



'''This builds the bidirectional RNN.  It only builds one layer of the RNN.
For a deeper model, stack BdRNNs together
    The basic idea is that we build two RNNs (left and right), and combine them
into a bdRNN.
    This function returns the outputs (the two RNNs concatenated) and the two states.
For classification purposes, the final state (outputs[-1] can be used as the predictor)

args: 
    x=The input tensor
    weight=A single weight set
    bias=a single bias set.  Set to zero if this is the final output
    
returns:
    outputs=Final output states of all timesteps
    SL=The final timestep output for the left side
    SR=The final timestep output for the right side
'''
def BdRNN(x, weight, bias, n_steps, n_hidden, n_input, name="output", reshape = False):

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    try:
        outputs, SL, SR = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)
    except Exception: #The older versions of TF do not produce the two output states
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        dtype=tf.float32)

    y = tf.add(tf.matmul(outputs[-1], weight), bias, name=name)

    return y, SL, SR


'''This function creates a single bdRNN.
args:
    n_steps=The number of timesteps
    n_hidden=The number of hidden nodes
    n_input=The size of the input
    
returns:
    X=The input tensor
    y=The label tensor
    pred=The output from the bdRNN

'''
def construct_automatically(n_steps, n_hidden, n_input,n_out, **kwargs):
    X = tf.placeholder("float", [None, n_steps, n_input], name='input_variable')
    y = tf.placeholder("float", [None, n_out], name='labels_variable')

    weights = tf.Variable(tf.random_normal([2 * n_hidden, n_out]))

    #The biases are optional.  Mainly because this first layer can be the output
    #layer
    if 'bias' in kwargs:
        if kwargs['bias']:
            bias = tf.Variable(tf.random_normal([n_input, n_steps]))
            pred, _, _ = BdRNN(X, weights, bias, n_steps, n_hidden)
    else:
        bias = tf.Variable(tf.random_normal([n_out]))
        pred, _, _ = BdRNN(X, weights, bias,n_steps, n_hidden, n_input)
    print "returned bdrnn: " + str(n_steps) + "  steps:input  " + str(n_input)
    return X, y, pred


def launch_mnist():
    idxs = [[range(0, 495), range(495, 550), range(550, 583)],
            [range(55, 550), range(0, 55), range(550, 583)],
            [range(0, 55) + range(110, 550), range(55, 110), range(550, 583)],
            [range(0, 110) + range(165, 550), range(110, 165), range(550, 583)],
            [range(0, 165) + range(220, 550), range(165, 220), range(550, 583)],
            [range(0, 220) + range(275, 550), range(220, 275), range(550, 583)],
            [range(0, 275) + range(330, 550), range(275, 330), range(550, 583)],
            [range(0, 330) + range(385, 550), range(330, 385), range(550, 583)],
            [range(0, 385) + range(440, 550), range(385, 440), range(550, 583)],
            [range(0, 440) + range(495, 550), range(440, 495), range(550, 583)]
            ]

    training_iters = 500
    batch_size = 20
    display_step = 10
    learning_rate = .001
    n_hidden_SNP = 10

    data_path_SNP = "./data_right.txt"
    labels_path_SNP = "./labels_right.txt"

    Pool_data_SNP = get_data(data_path_SNP)
    Pool_labels_SNP = make_one_hot(get_data(labels_path_SNP,type=np.int32),2)
    Pool_data_SNP = reorganize(Pool_data_SNP, 100)
    n_steps_SNP = Pool_data_SNP.shape[1]
    n_input_SNP = Pool_data_SNP.shape[2]

    OUTPUT = []
    fold_count = 0
    for idx in idxs:

        fold_output = []

        fold_data_train_SNP = Pool_data_SNP[idx[0]]
        fold_data_test_SNP = Pool_data_SNP[idx[1]]
        fold_data_valid_SNP = Pool_data_SNP[idx[2]]

        fold_labels_train = Pool_labels_SNP[idx[0]]
        fold_labels_test = Pool_labels_SNP[idx[1]]
        fold_labels_valid = Pool_labels_SNP[idx[2]]
        n_steps_SNP = fold_data_train_SNP.shape[1]

        X_SNP, y_SNP, pred_SNP = construct_automatically(n_steps_SNP, n_hidden_SNP, n_input_SNP, 400)

        #pred_SNP = tf.reshape(pred_SNP, [tf.shape(pred_SNP)[1],n_input_SNP*n_steps_SNP])

        X, y, pred = MLP.construct_automatically(400,[2], X=pred_SNP)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        predict = tf.equal(tf.argmax(pred, 1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(predict,tf.float32))

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            # Training cycle
            for epoch in range(training_iters):
                for j in range(1, len(fold_data_train_SNP), batch_size):
                    batch_x_MRI = fold_data_train_SNP[j:j + batch_size]

                    batch_y = fold_labels_train[j:j + batch_size]
                    sess.run(optimizer,
                             feed_dict={X_SNP: batch_x_MRI, y: batch_y})
                if epoch % display_step == 0:
                    train_accuracy = sess.run(accuracy,
                                              feed_dict={X_SNP: fold_data_train_SNP, y: fold_labels_train})
                    test_accuracy = sess.run(accuracy,
                                             feed_dict={X_SNP: fold_data_test_SNP, y: fold_labels_test})
                    print("fold: " + str(fold_count) + ", epoch: " + str(epoch) + "\n   train_accuracy: " + str(
                        train_accuracy) + ", test_accuracy: " + str(test_accuracy))
                    fold_output.append(test_accuracy)
        tf.reset_default_graph()
        OUTPUT.append(fold_output)
        fold_count += 1
        np.savetxt('SNP_BdRNN_only_NEW.txt',OUTPUT,delimiter=',')


if __name__ == '__main__':

    launch_mnist()