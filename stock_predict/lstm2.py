from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import tushare as ts


def getData(id, start, end, num):
    df = ts.get_hist_data(id, start, end)
    df = (df - np.sum(df) / len(df)) / (np.std(df))
    df = df[:num]
    df1 = np.array(df)
    # df2 = np.array(df.index)

    ##df = df.T
    x = []
    for i in range(len(df1)):
        # temp = np.append(df2[i],df1[i])
        temp = df1[i]
        newresult = []
        for item in temp:
            newresult.append(item)
        x.append(newresult)
    x.pop()
    x.reverse()
    return x


def getDataR(id, start, end, num):
    df = ts.get_hist_data(id, start, end)
    df = (df - np.sum(df) / len(df)) / (np.std(df))
    df = df[:num]
    df1 = np.array(df)
    # df2 = np.array(df.index)

    ##df = df.T
    x = []
    for i in range(len(df1)):
        # temp = np.append(df2[i],df1[i])
        temp = df1[i]
        newresult = []
        for item in temp:
            newresult.append(item)
        x.append(newresult)
    x.pop()

    P = df['close']
    templist = (P - P.shift(-5)) / P.shift(-5)
    tempDATA = []
    # 1  0.01  0
    ix = 0
    for indextemp in templist:
        if (ix % 5 == 0):
            if indextemp > 0:
                tempDATA.append([1, 0, 0])
            elif (indextemp <= 0):
                tempDATA.append([0, 1, 0])
            else:
                tempDATA.append([0, 0, 1])
            ix += 1
        else:
            ix += 1
    tempDATA.pop()
    y = tempDATA
    y.reverse()
    return y


df_sh = ts.get_sz50s()['code']
# train dataset
fac = []
ret = []
# test dataset
facT = []
retT = []
for ishare in df_sh:
    newfac = getData(ishare, '2008-07-22', '2016-07-22', 261)
    newret = getDataR(ishare, '2008-07-22', '2016-07-22', 261)
    # fac.append(newfac)
    for i in range(len(newfac)):
        fac.append(newfac[i])
    for i in range(len(newret)):
        ret.append(newret[i])

    newfacT = getData(ishare, '2016-08-01', '2017-01-10', 31)
    newretT = getDataR(ishare, '2016-08-01', '2017-01-10', 31)
    # fac.append(newfac)
    for i in range(len(newfacT)):
        facT.append(newfacT[i])
    for i in range(len(newretT)):
        retT.append(newretT[i])

newf = []
newfa = []
for i in range(len(fac)):
    if ((i + 1) % 5 != 0):
        newf.append(fac[i])
    else:
        newf.append(fac[i])
        newfa.append(newf)
        newf = []
fac = np.array(newfa)

ret = np.array(ret)

newfT = []
newfaT = []
for i in range(len(facT)):
    if ((i + 1) % 5 != 0):
        newfT.append(facT[i])
    else:
        newfT.append(facT[i])
        newfaT.append(newfT)
        newfT = []
facT = np.array(newfaT)

retT = np.array(retT)

print("**")
print(len(facT))
print(len(retT))
print("**")

fac = np.array(fac)
ret = np.array(ret)
learning_rate = 0.001
# Number of images entered into the model
batch_size = 10
training_iters = int(fac.shape[0] / batch_size)
display_step = 10

# Network Parameters
# Factors
n_input = 14
# time steps
n_steps = 5
n_hidden = 1024
# types of results
n_classes = 3

# tf Graph input
x = tf.placeholder('float', [None, n_steps, n_input])
y = tf.placeholder('float', [None, n_classes])

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_normal([2 * n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def BiRNN(x, weights, biases):
    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshape to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    try:
        outputs, _, _ = tf.nn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                dtype=tf.float32)
    except Exception:  # Old TensorFlow version only returns outputs not states
        outputs = tf.nn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                          dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


pred = BiRNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    for step in range(10):
        for i in range(int(len(fac) / batch_size)):
            batch_x = fac[i * batch_size:(i + 1) * batch_size].reshape([batch_size, n_steps, n_input])
            batch_y = ret[i * batch_size:(i + 1) * batch_size].reshape([batch_size, n_classes])
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            if i % display_step == 0:
                print(i, '----', (int(len(fac) / batch_size)))
        loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
        print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + "{:.6f}".format(
            loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
    print("Optimization Finished!")
    # Calculate accuracy for 128 mnist test images
    # test_len = 1280
    print("Accuracy in data set")
    test_data = fac[:batch_size].reshape([batch_size, n_steps, n_input])
    test_label = ret[:batch_size].reshape([batch_size, n_classes])
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

    print("Accuracy out of data set")
    test_dataT = facT[:batch_size].reshape([batch_size, n_steps, n_input])