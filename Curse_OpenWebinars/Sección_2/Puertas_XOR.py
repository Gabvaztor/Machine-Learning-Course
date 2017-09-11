import tensorflow as tf
from Curse_OpenWebinars.UsefulTools.TensorFlowUtils import initialize_session
import time

x_ = tf.placeholder(tf.float32, shape=[4, 2], name='x-input')
y_ = tf.placeholder(tf.float32, shape=[4, 1], name='y-input')

W1 = tf.Variable(tf.random_uniform(shape=[2, 2], minval=-1, maxval=1), name="W1")
W2 = tf.Variable(tf.random_uniform(shape=[2, 1], minval=-1, maxval=1), name="W2")

Bias1 = tf.Variable(tf.zeros([2]), name="Bias1")
Bias2 = tf.Variable(tf.zeros([1]), name="Bias2")

y1 = tf.sigmoid(tf.matmul(x_, W1) + Bias1)

y2 = tf.sigmoid(tf.matmul(y1, W2) + Bias2)

cost = tf.reduce_mean(((y_ * tf.log(y2)) + ((1 - y_) * tf.log(1.0 - y2))) * -1)

train_step = tf.train.GradientDescentOptimizer(0.03).minimize(cost)

XOR_X = [[0,0],[0,1],[1,0],[1,1]]
XOR_Y = [[0],[1],[1],[0]]

sess = initialize_session()

#writer = tf.summary.FileWriter("./logs/xor_logs", sess.graph_def)

t_start = time.clock()
for i in range(100000):
    sess.run(train_step, feed_dict={x_: XOR_X, y_: XOR_Y})
    if i % 1000 == 0:
        print('Epoch \n', i)
        print('W1 \n', sess.run(W1))
        print('Bias1 \n', sess.run(Bias1))
        print('W2 \n', sess.run(W2))
        print('Bias2 \n', sess.run(Bias2))
        print('y \n', sess.run(y2, feed_dict={x_: XOR_X, y_: XOR_Y}))
        print('cost \n', sess.run(cost, feed_dict={x_: XOR_X, y_: XOR_Y}))
t_end = time.clock()
print('Elapsed time ', t_end - t_start)