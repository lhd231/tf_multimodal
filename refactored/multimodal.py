import tensorflow as tf
import numpy as np
from utilities import *
import bRNN_basics_MNIST
import mlp_basics

import mdl_data

from tensorflow.contrib import rnn


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    print num_labels
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + np.ravel(labels_dense)] = 1
    return labels_one_hot

if __name__ == '__main__':
    data = mdl_data.YLIMED('./YLIMED_info.csv', "./img_audio_data/mfcc20",   "./img_audio_data/fc7")
    X_img_train = data.get_img_X_train()
    X_aud_train = data.get_aud_X_train()
    y_train = data.get_y_train()
    Y_train = dense_to_one_hot(y_train)
    # Shuffle initial data
    p = np.random.permutation(len(Y_train))
    X_img_train = X_img_train[p]
    X_aud_train = X_aud_train[p]
    Y_train = Y_train[p]

    # Load test data
    X_img_test = data.get_img_X_test()
    X_aud_test = data.get_aud_X_test()
    y_test = data.get_y_test()
    Y_test = dense_to_one_hot(y_test)

    learning_rate = 0.001
    training_epochs = 100
    batch_size = 256
    display_step = 1

    n_input_img = 4096  # YLI_MED image data input (data shape: 4096, fc7 layer output)
    n_hidden_1_img = 1000  # 1st layer num features 1000
    n_hidden_2_img = 600  # 2nd layer num features 600

    n_input_aud = 2000  # YLI_MED audio data input (data shape: 2000, mfcc output)
    n_hidden_1_aud = 1000  # 1st layer num features 1000
    n_hidden_2_aud = 600  # 2nd layer num features 600

    n_hidden_1_in = 600
    n_hidden_1_out = 256
    n_hidden_2_out = 128

    n_classes = 10  # YLI_MED total classes (0-9 digits)
    dropout = 0.75
    X_img, y_img, output_img = mlp_basics.construct_automatically(n_input_img,[n_hidden_1_img,n_hidden_2_img])

    X_aud, y_aud, output_aud = mlp_basics.construct_automatically(n_input_aud,[n_hidden_1_aud,n_hidden_2_aud, n_classes], dropout=dropout)

    added = tf.add(output_img, output_aud)

    _, y_comb, output_comb = mlp_basics.construct_automatically(n_hidden_1_in, [n_hidden_1_out, n_hidden_2_out],X=added)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_comb, labels=y_comb))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        sess.run(init)
        #Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(len(Y_train)/batch_size)
            #Loop oveer all batches
            for i in range(total_batch):
                batch_x_aud, batch_x_img, batch_ys, finish = data.next_batch_multi(X_aud_train, X_img_train, Y_train, batch_size, len(Y_train))
                # Fit traning using batch data
                sess.run(optimizer, feed_dict = {X_aud: batch_x_aud, X_img: batch_x_img, y_comb: batch_ys})
                # Compute average loss
                avg_cost += sess.run(cost, feed_dict = {X_aud: batch_x_aud, X_img: batch_x_img, y_comb: batch_ys}) / total_batch
                #Shuffling
                if finish:
                    p = np.random.permutation(len(Y_train))
                    X_aud_train = X_aud_train[p]
                    X_img_train = X_img_train[p]
                    Y_train = Y_train[p]
            # Display logs per epoch step
            if epoch % display_step == 0:
                print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)
        print "Optimization Finished!"

        # Test model
        correct_prediction = tf.equal(tf.argmax(output_comb, 1), tf.argmax(y_comb, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print "Accuracy:", accuracy.eval({X_aud: X_aud_test, X_img: X_img_test, y_comb: Y_test, keep_prob: 1.})
        print 'MM.py'
