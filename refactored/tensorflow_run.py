# Implementation of a simple MLP network with one hidden layer. Tested on the iris data set.
# Requires: numpy, sklearn>=0.18.1, tensorflow>=1.0

# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2
import tensorflow as tf
import numpy as np
import scipy.io

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

learning_rate = 0.001
training_epochs = 150
batch_size = 100
display_step = 1

n_hidden_1 = 5000 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 96926 # MNIST data input (img shape: 28*28)
n_classes = 2 # MNIST total classes (0-9 digits)
# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    layer_1 = tf.layers.dropout(layer_1,rate=.5)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    layer_2 = tf.layers.dropout(layer_2,rate=.5)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}



def make_one_hot(target,labels):
    targets = np.zeros((len(target),labels))
    targets[np.arange(len(target)),target] = 1
    return targets

def get_data_mri(seed=15,t=160):
    print('entered get data')
    data   = np.loadtxt('/export/mialab/users/dlin/MCIC/ToNoah/data_mri.csv',delimiter=',')
    print('loaded data')
    target = np.loadtxt('/export/mialab/users/dlin/MCIC/ToNoah/labels_mri.csv',delimiter=',',dtype=np.int32)
    print('loaded label')
    print data.shape
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
    targets = make_one_hot(target,num_labels)

    train_t = targets[:t]
    test_t = targets[t:]
    print('returning data')
    print train_t.shape
    print train_d.shape
    print train_t[:20]
    return train_d,test_d,train_t,test_t
def get_data_SNPS():
    data = scipy.io.loadmat('/export/mialab/users/dlin/MCIC/ToNoah/cobre_mcic_data.mat')['E']
    labels = scipy.io.loadmat('/export/mialab/users/dlin/MCIC/ToNoah/labels_combSet.mat')['labels']

    N, M = data.shape
    all_X = np.ones((N,M+1))
    all_X[:,1:] = data


def main():
    train_size = 160
    train_X, test_X, train_y, test_y = get_data_mri(t=train_size)


    # Construct model
    pred = multilayer_perceptron(x, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        batch_size = 20
        total_batch = train_size/batch_size
        for epoch in range(training_epochs):
            avg_cost = 0.
            # Loop over all batches
            for i in range(total_batch):
                batch_x = train_X[i:batch_size]
                batch_y = train_y[i:batch_size]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                             y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
                # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                  "{:.9f}".format(avg_cost))
        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: test_X, y: test_y}))




if __name__ == '__main__':
    main()