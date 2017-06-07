from __future__ import print_function

import tensorflow as tf
from utilities import *
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

'''This file creates an MLP model with given parameters.  The functions can be
imported to be used at your leisure.  Or, if the file is run as a main, it builds
an MLP and uses the MNIST dataset to test the model
'''


'''This returns the model.  It's the set of all functions for the forward pass
If you already have the input variable and the weights or biases, this function
can be called manually
args:
    x=Input variable
    weights=A weight set for the entire model
    biases=The biases for the model
    dropout=A scalar representing the dropout percentage (optional)
output:
    The final output after all forward pass functions
To note:  matmul only works on 2D tensors
'''
def multilayer_perceptron(x, weights, biases, dropout=None):
    for weight,bias in zip(weights,biases):
        x = tf.add(tf.matmul(x, weight), bias)
        if dropout != None:
            x = tf.nn.dropout(x, dropout)
    return x


'''This will construct the weights and biases automatically.  As well as the input
and output tensors.  Then, it builds the functions through the multilayer_perceptron
function
args:
    input_size=The size of the input
    layer_sizes=list of layer sizes (including output size)
    dropout=A single value representing percentage
    X=An optional place to add your own input tensor
output:
    X=The input variable
    y=the label variable
    pred=Model output, or prediction variable
'''
def construct_automatically(input_size,layer_sizes, dropout=None, X=None):
    weights = []
    biases = []

    #These are the input and output tensors
    if X==None:
        X = tf.placeholder("float", [None, input_size])
    else:
        x = X
    y = tf.placeholder("float", [None, layer_sizes[-1]])

    #These are the weights and biases for the hidden layers
    for i,layer in enumerate(layer_sizes):
        weights.append(tf.Variable(tf.random_normal([input_size,layer])))
        biases.append(tf.Variable(tf.random_normal([layer])))
        input_size = layer


    #Here we build the functions
    pred = multilayer_perceptron(X,weights,biases,dropout)
    return X, y, pred


'''This will create a session and run it using the MNIST dataset
NOTES:
    The session is what starts and guides the computation.
Before this step, the graph has already been built and consists of all of the
operations you previously made.  But, when Session.run() is called, it places 
this graph on the GPU.  The 'init' variable is an initializer op that 
initializes all of the variables in the graph.

    The subsequent sess.run() calls are used for specific operations.  In this
case, by using the optimizer and cost functions, we're calling every op 
from that part of the graph.  So, it can be used for a single operation, such
as a single matmul, or, as it is in this case, the forward pass for the model
as well as the backpropagation.
The docs for this are found here:
    https://www.tensorflow.org/api_guides/python/train
All optimizers use the minimize() function to minimize some error function.  
This minimize function also calls compute_gradients() and apply_gradients().

    In this example, sess.run(...) returns the output from calls to both the
optimizer and cost.  Only the output from the cost function is printed.  
'''
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

    data_path_MRI = './data.csv'
    labels_path_MRI = './labels.csv'

    Pool_data_MRI = get_data(data_path_MRI)
    Pool_labels_MRI = get_data(labels_path_MRI)
    n_input = Pool_data_MRI.shape[1]

    training_iters = 500
    batch_size = 20
    display_step = 10
    learning_rate = .01

    n_classes = 2

    n_hidden_MRI_1 = 3000
    n_hidden_MRI_2 = 1000
    n_hidden_MRI_3 = 300

    dropout = .5

    OUTPUT = []
    fold_count = 0
    for idx in idxs:
        X, y, pred = construct_automatically(n_input,[n_hidden_MRI_1,n_hidden_MRI_2, n_hidden_MRI_3,n_classes], dropout=dropout)

        fold_output = []

        fold_data_train_MRI = Pool_data_MRI[idx[0]]
        fold_data_test_MRI = Pool_data_MRI[idx[1]]
        fold_data_valid_MRI = Pool_data_MRI[idx[2]]

        fold_labels_train = Pool_labels_MRI[idx[0]]
        fold_labels_test = Pool_labels_MRI[idx[1]]
        fold_labels_valid = Pool_labels_MRI[idx[2]]

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        predict = tf.equal(tf.argmax(pred, 1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(predict,tf.float32))

        # Initializing the global variables (this gives a default graph)
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            # Training cycle
            for epoch in range(training_iters):
                for j in range(1, len(fold_data_train_MRI), batch_size):
                    batch_x_MRI = fold_data_train_MRI[j:j + batch_size]

                    batch_y = fold_labels_train[j:j + batch_size]
                    sess.run(optimizer,
                             feed_dict={X: batch_x_MRI, y: batch_y})
                if epoch % display_step == 0:
                    train_accuracy = sess.run(accuracy,
                                              feed_dict={X: fold_data_train_MRI, y: fold_labels_train})
                    test_accuracy = sess.run(accuracy,
                                             feed_dict={X: fold_data_test_MRI, y: fold_labels_test})
                    print("fold: " + str(fold_count) + ", epoch: " + str(epoch) + "\n   train_accuracy: " + str(
                        train_accuracy) + ", test_accuracy: " + str(test_accuracy))
                    fold_output.append(test_accuracy)
        tf.reset_default_graph()
        OUTPUT.append(fold_output)
        fold_count += 1
        np.savetxt('MRI_only.txt',OUTPUT,delimiter=',')


if __name__ == '__main__':

    launch_mnist()
