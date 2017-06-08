import tensorflow as tf
import numpy as np
from utilities import *
import BdRNN_finalOutput
import time
import MLP
import os

from img_audio_data import mdl_data
from tensorflow.contrib import rnn

'''NOTES:  This uses two datasets from:
    http://multimedia-commons.s3-website-us-west-2.amazonaws.com/?prefix=subsets/YLI-MED/
     - subsets/YLI-MED/features/audio/mfcc20/mfcc20.tgz
     - subsets/YLI-MED/features/keyframe/alexnet/fc7.tgz
The code to parse this information is from: 
    https://github.com/lheadjh/MultimodalDeepLearning
        - The img_audio_data/mdl_data.py is the file that does most of the work
        to parse the data
'''

'''This was taken from the aforementioned code base.  But, it's a simple make_one_hot
function
'''
def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    print num_labels
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + np.ravel(labels_dense)] = 1
    return labels_one_hot

if __name__ == '__main__':
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
    learning_rate = .01

    data_path_SNP = "./data_right.txt"
    labels_path_SNP = "./labels_right.txt"

    data_path_MRI = './data.csv'
    labels_path_MRI = './labels.csv'



    n_hidden_SNP = 10

    #NOTES

    n_hidden_MRI_1 = 3000
    n_hidden_MRI_2 = 1000
    n_hidden_MRI_3 = 300

    dropout = .5

    Pool_data_SNP = get_data(data_path_SNP)
    Pool_labels_SNP = get_data(labels_path_MRI)
    Pool_data_SNP = reorganize(Pool_data_SNP, 100)
    n_steps_SNP = Pool_data_SNP.shape[1]
    n_input_SNP = Pool_data_SNP.shape[2]

    Pool_data_MRI = get_data(data_path_MRI)
    Pool_labels_MRI = get_data(labels_path_MRI)
    n_input_MRI = Pool_data_MRI.shape[1]


    
    #folder = "/export/mialab/users/nlewis/tf_multimodal/pretrain_model_UPENN_" + "batch:_" + str(
    #    batch_size) + "_hidden_" + str(n_hidden) + "_" + str(learning_rate)
    #name = folder + "/model.ckpt"
    file_name = 'multimodal_output/MULTIMODAL_'+time.strftime("%d_%m:%H:%M")+'_dropout:_'+str(dropout)+'_FIRSTHALF.txt'
    OUTPUT = []
    for i in range(0,len(idxs)):
        fold_output = []

        fold_data_train_SNP = Pool_data_SNP[idxs[i][0]]
        fold_data_test_SNP = Pool_data_SNP[idxs[i][1]]
        fold_data_valid_SNP = Pool_data_SNP[idxs[i][2]]

        fold_data_train_MRI = Pool_data_MRI[idxs[i][0]]
        fold_data_test_MRI = Pool_data_MRI[idxs[i][1]]
        fold_data_valid_MRI = Pool_data_MRI[idxs[i][2]]

        fold_labels_train = Pool_labels_SNP[idxs[i][0]]
        fold_labels_test = Pool_labels_SNP[idxs[i][1]]
        fold_labels_valid = Pool_labels_SNP[idxs[i][2]]
        n_steps_SNP = fold_data_train_SNP.shape[1]

        X_SNP, y_SNP, pred_SNP = BdRNN_finalOutput.construct_automatically(n_steps_SNP, n_hidden_SNP, n_input_SNP, 300, reshape=True)

        #pred_SNP = tf.reshape(pred_SNP, [tf.shape(pred_SNP)[1],n_input_SNP*n_steps_SNP])
        X_MRI, y_MRI, pred_MRI = MLP.construct_automatically(n_input_MRI,[n_hidden_MRI_1,n_hidden_MRI_2, n_hidden_MRI_3],dropout=dropout,input_name='absolute_input')

        conc = tf.concat([pred_SNP,pred_MRI], axis=1)
        print n_steps_SNP
        print n_input_SNP



        x_COMB, y_COMB, pred_COMB = MLP.construct_automatically(300 + 300,[400,2], X=conc, name='absolute_output')
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_COMB, labels=y_COMB))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        predict = tf.equal(tf.argmax(pred_COMB, 1), tf.argmax(y_COMB,1))
        accuracy = tf.reduce_mean(tf.cast(predict,tf.float32))

        init = tf.global_variables_initializer()
        folder = "/export/mialab/users/nlewis/tf_multimodal/pretrain_models/outpu_model_fusion_" + time.strftime(
            "%d_%m:%H:%M") +"_fold:_"+str(i)+ "_batch:_" + str(batch_size) + "_hidden_" + str(n_hidden_MRI_1) + "_" + str(learning_rate)
        os.makedirs(folder)
        name_model = folder + "/model.ckpt"

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            cur_batch_size = batch_size
            for epoch in range(training_iters):
                for j in range(1, len(fold_data_train_SNP), batch_size):
                    batch_x_SNP = fold_data_train_SNP[j:j + cur_batch_size]
                    batch_x_MRI = fold_data_train_MRI[j:j + cur_batch_size]

                    batch_y = fold_labels_train[j:j + cur_batch_size]
                    sess.run(optimizer,
                             feed_dict={X_MRI: batch_x_MRI, X_SNP: batch_x_SNP,
                                        y_COMB: batch_y})
                if epoch % display_step == 0:
                    train_accuracy = sess.run(accuracy,
                                                      feed_dict={X_MRI: fold_data_train_MRI, X_SNP: fold_data_train_SNP, y_COMB: fold_labels_train})
                    test_accuracy = sess.run(accuracy,
                                                      feed_dict={X_MRI: fold_data_test_MRI, X_SNP: fold_data_test_SNP, y_COMB: fold_labels_test})
                    print("fold: "+str(i)+", epoch: "+str(epoch)+"\n   train_accuracy: " + str(train_accuracy) + ", test_accuracy: "+str(test_accuracy))
                    fold_output.append(test_accuracy)

            saver.save(sess, name_model)
        OUTPUT.append(fold_output)
        np.savetxt(file_name,OUTPUT,delimiter=',')
        tf.reset_default_graph()

np.savetxt(file_name,OUTPUT,delimiter=',')

