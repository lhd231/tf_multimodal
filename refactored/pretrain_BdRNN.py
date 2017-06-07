import tensorflow as tf
import numpy as np
from utilities import *
from BdRNN import *
import os
import time

from tensorflow.contrib import rnn

'''This pretrains a model using the UPENN dataset.  The output location
For the model details is hardcoded.  Change as suits your needs
'''
training_iters = 700
batch_size = 300
display_step = 10

n_hidden = 50
learning_rate = .01

data_path = ".././genodata_imputed_upenn_selected.txt"

Pool_data = get_data(data_path,delim= ' ')

Pool_data = reorganize(Pool_data,100)
n_steps = Pool_data.shape[1]
n_input = Pool_data.shape[2]

X, y, pred = construct_automatically(n_steps, n_hidden, n_input)
cost = tf.reduce_mean(tf.pow(y-pred,2))
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

error = tf.reduce_mean(tf.abs(y-pred))
init = tf.global_variables_initializer()

#This is hardcoded for my own purposes.  You should change it if
#you wish to run this file
folder = "/export/mialab/users/nlewis/tf_multimodal/pretrain_models/pretrain_model_UPENN_"+time.strftime("%d_%m:%H:%M")+"_batch:_"+str(batch_size) +"_hidden_"+ str(n_hidden) + "_" + str(learning_rate)
os.makedirs(folder)
name = folder + "/model.ckpt"

saver = tf.train.Saver()

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
            acc = sess.run(error,
                           feed_dict={X: Pool_data, y: Pool_data.reshape(n_steps, Pool_data.shape[0], n_input)})
            saver.save(sess,name)
            print(" Iter: " + str(epoch) + ", Training Cost= " + \
                  "{:.5f}".format(acc)+" \n")
    saver.save(sess,name)



