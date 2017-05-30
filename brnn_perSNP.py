
display_step = 10

n_input = 3
n_steps = 100
learning_rate = 0.001
training_iters = 200
batch_size = 20
n_hidden = 1

def get_data(datafile, labelfile,seed=21,t=500,delim=',',ty=np.float32):
    print('entered get data')
    data   = np.loadtxt(datafile,delimiter=delim,dtype=np.float32)
    print('loaded data')
    target = np.loadtxt(labelfile,delimiter=delim,dtype=np.int32)
    print('loaded label')
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
    print('returning data')
    return data,target

def BiRNN(x, weights, biases):

    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Define lstm cells with tensorflow
    # Forward direction cell
    print("in birnn " + str(n_hidden))
    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    print("made left and right")
    # Get lstm cell output
    try:
        outputs, SL, SR = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    target = tf.add(SL,SR)
    return tf.matmul(target, weights['out'])# + biases['out']

def make_one_hot(target):
    targets = (np.arange(target.max()+1) == target[:,:,None]).astype(int)
    return targets

def run(data_path = "./data_right.txt",labels_path="./labels_right.txt",delim=','):
    global n_steps, n_hidden, n_input, display_step,batch_size,learning_rate,training_iters
    Pool_data, Pool_labels = get_data(data_path,labels_path, delim=delim,ty=np.int32)
    OUTPUT = []
    Pool_data = make_one_hot(Pool_data)
    Pool_labels = make_one_hot_2(Pool_labels,2)
    total_size = Pool_data.shape[1]
    print(total_size)
    print(n_steps)
    n_input = Pool_data.shape[2] *(total_size - (Pool_data.shape[1] % n_steps)) / n_steps
    print(n_input)
    #n_steps = Pool_data.shape[1]
    for i in range(len(idxs)):
        print(n_hidden)
        print(n_steps)
        tf.logging.set_verbosity(tf.logging.ERROR)
        weights = {
            # Hidden layer weights => 2*n_hidden because of forward + backward cells
            'out': tf.Variable(tf.random_normal([n_steps,2 * n_hidden,n_input]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([n_steps]))
        }
        x = tf.placeholder("float", [None,n_steps,n_input])
        I = tf.Variable(tf.random_normal([100, 2]))
        y = tf.placeholder("float", [None, 2])
        SNP_label_train = Pool_labels[idxs[i][0]]
        SNP_label_test = Pool_labels[idxs[i][1]]
        SNP_data_train,SNP_data_test = organize_data([Pool_data[idxs[i][0]],Pool_data[idxs[i][1]]],n_steps)
        print('got data and organized')
        pred = BiRNN(x, weights, biases)
        print("break")
        print(SNP_data_test.shape)
        print(SNP_data_train.shape)
        reshape = tf.reshape(pred,[tf.shape(pred)[1],n_steps,n_input])
        flat = tf.reduce_mean(reshape,2)
        #SNP_label_train = make_one_hot(Pool_labels[idxs[i][0]], 2)
        #SNP_label_test = make_one_hot(Pool_labels[idxs[i][1]], 2)
        # Define loss and optimizer
        out = tf.matmul(flat,I)
        cost = tf.square(tf.reduce_mean((y-out)))
        print('have cost')
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        print('got optimizer')
        # Evaluate model
        #correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))

        accuracy = cost
        # Initializing the variables
        init = tf.global_variables_initializer()
        fold = []
        # Launch the graph
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            sess.run(init)
            step = 1
            # Keep training until reach max iterations

            for epoch in range(training_iters):
                for i in range(1,len(SNP_data_train),batch_size):
                    batch_x= SNP_data_train[i:i+batch_size]
                    batch_y= SNP_label_train[i:i+batch_size]
                    #batch_x.reshape(n_steps,cur_size,n_input
                    cur_size=batch_x.shape[0]
                    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                # Reshape data to get 28 seq of 28 elements
                #batch_x = batch_x.reshape((batch_size, n_steps, n_input))
                # Run optimization op (backprop)
                if epoch % display_step == 0:
                    # Calculate batch accuracy
                    #SNP_data_train.reshape(n_steps,495,n_input)
                    acc = sess.run(accuracy, feed_dict={x: SNP_data_train, y: SNP_label_train})
                    # Calculate batch loss
                    #batch_x.reshape(n_steps,cur_size,n_input)
                    #SNP_data_test.reshape(n_steps,55,n_input)
                    #loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                    print("Iter " + str(epoch) +  ", Training Cost= " + \
                          "{:.5f}".format(acc))
                    test_acc = sess.run(accuracy, feed_dict={x: SNP_data_test, y: SNP_label_test})
                    print("Testing Cost: " + str(test_acc))


                    fold.append(test_acc)
            print("Optimization Finished!")
            OUTPUT.append(fold)
            # Calculate accuracy for 128 mnist test images
            test_len = SNP_data_test.shape[0]
            test_data = SNP_data_test#mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
            #test_label = SNP_label_test#mnist.test.labels[:test_len]
            #print("Testing Accuracy:", \
            #    sess.run(accuracy, feed_dict={x: test_data, y: test_data}))
        tf.reset_default_graph()
    np.savetxt('output_results_from_BRNN.txt',OUTPUT,delimiter=',')

run()