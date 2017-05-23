import numpy as np
import tensorflow as tf
import os

def forwardprop(X, weights,biases,dropout=None):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    #output = weights.pop(-1)
    h = tf.nn.sigmoid(tf.add(tf.matmul(X,weights.pop(0)),biases.pop(0)))
    print "forward " + str(len(weights))
    for w in range(len(weights)):
        h = tf.nn.sigmoid(tf.add(tf.matmul(h,weights[w]),biases[w]))
        if dropout != None:
            h = tf.nn.dropout(h,dropout)
     # The \varphi function
    return h

def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    biases = tf.random_normal([shape[1]])
    return tf.Variable(weights), tf.Variable(biases)


def get_data(datafile, seed=21,t=500,delim=','):
    print('entered get data')
    data   = np.loadtxt(datafile,delimiter=delim,dtype=np.float32)
    print('loaded data')
    print data.shape
    #target = np.loadtxt(labelfile,delimiter=delim,dtype=np.int32)
    #print('loaded label')
    #print target.shape
    # Prepend the column of 1s for bias
    N, M  = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data
    # Convert into one-hot vectors
    #num_labels = len(np.unique(target))
    #all_Y = np.eye(num_labels)[target]  # One liner trick!
    np.random.seed(seed)
    p = np.random.permutation(N)
    data = data[p]
    #target = target[p]
    train_d = data[:t]
    test_d = data[t:]
    #targets = make_one_hot(target,num_labels)
    #targets = target
    #train_t = targets[:t]
    #test_t = targets[t:]
    print('returning data')
    #print train_t.shape
    print test_d.shape

    return train_d,test_d#,train_t,test_t

def main():
    h_layer = 1000
    h_decoder_layer = 1000
    h_middle_layer = 300
    folder = "/home/lhd/tensorflow/For_Tensorflow_Multimodal/model_"+str(h_layer)+"_"+str(h_middle_layer)+"_"+str(h_decoder_layer)
    #os.makedirs(folder)
    name = folder +"/model.ckpt"
    X_train,X_test = get_data('/home/lhd/upenn/genodata_imputed_upenn_selected.txt',delim=' ',t=6500)
    input_size = X_train.shape[1]
    X = tf.placeholder("float", shape=[None, input_size],name='X')
    weights = tf.random_normal((input_size,h_layer), stddev=0.1)
    biases = tf.random_normal([h_layer])
    weights2 = tf.random_normal((h_layer,h_middle_layer), stddev=0.1)
    biases2 = tf.random_normal([h_middle_layer])
    weights3 = tf.random_normal((h_middle_layer,h_decoder_layer), stddev=0.1)
    biases3 = tf.random_normal([h_decoder_layer])
    weights4 = tf.random_normal((h_decoder_layer,input_size))
    biases4 = tf.random_normal([input_size])
    W1 = tf.Variable(weights,name='W1')
    B1 = tf.Variable(biases,name='B1')
    W2 = tf.Variable(weights2,name='W2')
    B2 = tf.Variable(biases2,name='B2')
    W3 = tf.Variable(weights3,name='W3')
    B3 = tf.Variable(biases3,name='B3')
    W4 = tf.Variable(weights4,name='W4')
    B4 = tf.Variable(biases4,name='B4')

    forward_pass = forwardprop(X, [W1,W2,W3],[B1,B2,B3],dropout=.3)

    yhat = tf.nn.sigmoid(tf.add(tf.matmul(forward_pass, W4),B4))
    #predict = tf.argmax(yhat, axis=1)
    Y = X
    # Backward propagation
    cost    = tf.reduce_mean(tf.pow(Y-yhat,2))

    updates = tf.train.RMSPropOptimizer(.001).minimize(cost)
    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    init = tf.global_variables_initializer()
    sess.run(init)
    batch_size = 20
    for epoch in range(1):
        # Train with each example
        for i in range(0,len(X_train),batch_size):
            _,c = sess.run([updates,cost],
                     feed_dict={X: X_train[i: i + batch_size]})

        #train_accuracy = np.mean(np.argmax(X_train, axis=1) ==
        #                         sess.run([updates,cost], feed_dict={X: X_train}))
        #test_accuracy = np.mean(np.argmax(X_test, axis=1) ==
        #                        sess.run([updates,cost], feed_dict={X: X_test}))

        print("Epoch = %d, train accuracy = %.2f%%"
              % (epoch + 1, c))
        if epoch%10 == 0:
            saver.save(sess,name)
    saver.save(sess,name)
    sess.close()


if __name__ == '__main__':
    main()
