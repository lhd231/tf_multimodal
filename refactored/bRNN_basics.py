import tensorflow as tf
import numpy as np
from utilities import *
from bRNN_basics_MNIST import *

from tensorflow.contrib import rnn





data_path = "./data_right.txt"
labels_path="./labels_right.txt"

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

training_iters = 200
batch_size = 25
display_step = 10
classes = 3
n_hidden = 10
Pool_data = get_data(data_path)
Pool_labels = get_data(labels_path)
Pool_data = reorganize(Pool_data,100)
n_steps = Pool_data.shape[1]
n_input = Pool_data.shape[2]

for i in range(len(idxs)):
    fold_train = Pool_data[idxs[i][0]]
    fold_test = Pool_data[idxs[i][1]]

    n_steps = fold_train.shape[1]
    X, y, pred = construct_automatically(n_steps, n_hidden, n_input)

    cost = tf.reduce_mean(tf.abs(y - pred))
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(init)
        cur_batch_size = batch_size
        for epoch in range(training_iters):
            for i in range(1, len(fold_train), batch_size):
                batch_x = fold_train[i:i + cur_batch_size]
                batch_y = fold_train[i:i + cur_batch_size].reshape(n_steps,cur_batch_size,n_input)
                sess.run(optimizer, feed_dict={X: batch_x, y: batch_y})
                cur_batch_size = fold_train[i+batch_size:i + 2*batch_size].shape[0]

            if epoch % display_step == 0:
                acc = sess.run(cost, feed_dict={X: fold_train, y: fold_train.reshape(n_steps,fold_train.shape[0],n_input)})
                test_acc = sess.run(cost, feed_dict={X: fold_test, y: fold_test.reshape(n_steps,fold_test.shape[0],n_input)})

                print(" Iter: " + str(epoch) + ", Training Cost= " + \
                      "{:.5f}".format(acc/2) + ", Testing Cost: " + str(test_acc/2) + " \n")

        valid = sess.run(cost, feed_dict={X: fold_valid, y: fold_valid.reshape(n_steps, fold_valid.shape[0], n_input)})
        print("Validation accuracy: " + str(valid/2))

    tf.reset_default_graph()
    exit()
print("Optimization Finished!")
