import tensorflow as tf
import numpy as np
from utilities import *
from bRNN_basics_MNIST import *

from tensorflow.contrib import rnn

training_iters = 150
batch_size = 50
display_step = 10

n_hidden = 10

data_path = ".././genodata_imputed_upenn_selected.txt"

Pool_data = get_data(data_path,delim= ' ')

Pool_data = reorganize(Pool_data,100)
n_steps = Pool_data.shape[1]
n_input = Pool_data.shape[2]

X, y, pred = construct_automatically(n_steps, n_hidden, n_input)

cost = tf.reduce_mean(tf.abs(y - pred))
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(init)
    cur_batch_size = batch_size
    for epoch in range(training_iters):
        for i in range(1, len(Pool_data), batch_size):
            batch_x = Pool_data[i:i + cur_batch_size]
            batch_y = Pool_data[i:i + cur_batch_size].reshape(n_steps, cur_batch_size, n_input)
            sess.run(optimizer, feed_dict={X: batch_x, y: batch_y})
            cur_batch_size = Pool_data[i + batch_size:i + 2 * batch_size].shape[0]

        if epoch % display_step == 0:
            acc = sess.run(cost,
                           feed_dict={X: Pool_data, y: Pool_data.reshape(n_steps, Pool_data.shape[0], n_input)})

            print(" Iter: " + str(epoch) + ", Training Cost= " + \
                  "{:.5f}".format(acc / 2)+" \n")


