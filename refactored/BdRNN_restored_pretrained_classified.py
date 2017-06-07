import tensorflow as tf
import numpy as np
from utilities import *

from BdRNN import *
import time




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
training_iters = 400
batch_size = 100
display_step = 10

n_hidden = 50
learning_rate = .01

Pool_data = get_data(data_path)
Pool_labels = make_one_hot(get_data(labels_path,type=np.int32),2)
Pool_data = reorganize(Pool_data,100)
n_steps = Pool_data.shape[1]
n_input = Pool_data.shape[2]

folder = "/export/mialab/users/nlewis/tf_multimodal/pretrain_models/pretrain_model_UPENN_06_06:15:11_batch:_500_hidden_2_0.01"

name = folder + "/model.ckpt"

OUTPUT = []
file_name = "BdRNN_classified_"+time.strftime("%d_%m:%H:%M")+"_n_hidden:_"+str(n_hidden)+"_learn:_"+str(learning_rate)+".txt"
for i in range(len(idxs)):
    fold_output = []

    fold_data_train = Pool_data[idxs[i][0]]
    fold_data_test = Pool_data[idxs[i][1]]
    fold_data_valid = Pool_data[idxs[i][2]]

    fold_labels_train = Pool_labels[idxs[i][0]]
    fold_labels_test = Pool_labels[idxs[i][1]]
    fold_labels_valid = Pool_labels[idxs[i][2]]

    n_steps = fold_data_train.shape[1]
    X_SNP, y_SNP, pred_SNP = construct_automatically(n_steps, n_hidden, n_input)

    I = tf.Variable(tf.random_normal([n_input, 2]))
    y = tf.placeholder("float", [None, 2])

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        saver = tf.train.import_meta_graph(name + ".meta")
        saver.restore(sess, name)
        #saver.restore(sess, tf.train.latest_checkpoint("/home/lhd/tensorflow/For_Tensorflow_Multimodal/model_"+str(h_layer)+"_"+str(h_middle_layer)+"_"+str(h_decoder_layer)+'/'))
        Ops = sess.graph.get_operations()

        X = sess.graph.get_tensor_by_name("input_variable:0")
        pred_SNP = sess.graph.get_tensor_by_name("output:0")

        #reshape = tf.reshape(pred_SNP, [tf.shape(pred_SNP)[1], n_steps * n_input])

        pred = tf.matmul(pred_SNP[-1], I)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

        predict = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))

        sess.run(init)
        cur_batch_size = batch_size

        for epoch in range(training_iters):
            for i in range(1, len(fold_data_train), batch_size):
                batch_x = fold_data_train[i:i + cur_batch_size]
                batch_y = fold_labels_train[i:i + cur_batch_size]
                sess.run(optimizer, feed_dict={X: batch_x, y: batch_y})
                cur_batch_size = fold_data_train[i + batch_size:i + 2 * batch_size].shape[0]

            if epoch % display_step == 0:
                test_acc = sess.run(accuracy,
                                    feed_dict={X: fold_data_test, y: fold_labels_test})

                print(" Iter: " + str(epoch)  + ", Testing Cost: " + str(test_acc) + " \n")
                fold_output.append(test_acc)
        #valid = sess.run(cost, feed_dict={X: fold_valid, y: fold_valid.reshape(n_steps, fold_valid.shape[0], n_input)})
        #print("Validation accuracy: " + str(valid / 2))

        OUTPUT.append(fold_output)
        np.savetxt(file_name,OUTPUT,delimiter=',')
    tf.reset_default_graph()
np.savetxt('final_out_pretrained_bdrnn.txt',OUTPUT,delimiter=',')
print "complete"
