import tensorflow as tf
from Course_OpenWebinars.UsefulTools.TensorFlowUtils import initialize_session
import matplotlib.pyplot as plt
import time

x_ = tf.placeholder(tf.float32, shape=[4, 2], name='x-input')
y_ = tf.placeholder(tf.float32, shape=[4, 1], name='y-input')

W1 = tf.Variable(tf.random_uniform(shape=[2, 2], minval=-0.5, maxval=0.5), name="W1")
W2 = tf.Variable(tf.random_uniform(shape=[2, 1], minval=-0.5, maxval=0.5), name="W2")

Bias1 = tf.Variable(tf.zeros([2]), name="Bias1")
Bias2 = tf.Variable(tf.zeros([1]), name="Bias2")

# En este caso es necesaria la función de activación sigmoide porque los valores deben estar comprendidos entre 0 y 1.
y1 = tf.sigmoid(tf.matmul(x_, W1) + Bias1)
y2 = tf.sigmoid(tf.matmul(y1, W2) + Bias2)

cost = tf.reduce_mean(((y_ * tf.log(y2)) + ((1 - y_) * tf.log(1.0 - y2))) * -1)
#cost = tf.reduce_mean(tf.abs(tf.subtract(y_, y2)))

train_step = tf.train.AdamOptimizer(0.002).minimize(cost)
#train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cost)

XOR_X = [[0,0],[0,1],[1,0],[1,1]]
XOR_Y = [[0],[1],[1],[0]]

sess = initialize_session()

t_start = time.clock()
y0_ = []
y1_ = []
y2_ = []
y3_ = []
costs = []

# Puede ocurrir que el descenso del gradiente quede en un mínimo local disminuyendo una o dos de las variables
# y dejando las otras igual. Ese es un problema relativo al valor inicial de los pesos y de los biases (por ser random)
# Para solucionarlo deberíamos elegir un valor concreto para las variables que encamine al aprendizaje.
# Ya se verá más adelante durante el curso.
for i in range(100000):
    sess.run(train_step, feed_dict={x_: XOR_X, y_: XOR_Y})
    y_prediction = y2.eval(feed_dict={x_: XOR_X, y_: XOR_Y})
    coste = cost.eval(feed_dict={x_: XOR_X, y_: XOR_Y})
    if i % 2 == 0:
        y0_.append(y_prediction[0])
        y1_.append(y_prediction[1])
        y2_.append(y_prediction[2])
        y3_.append(y_prediction[3])
        costs.append(coste)
    if i % 1000 == 0:
        print('Epoch \n', i)
        print('W1 \n', sess.run(W1))
        print('Bias1 \n', sess.run(Bias1))
        print('W2 \n', sess.run(W2))
        print('Bias2 \n', sess.run(Bias2))
        print('y \n', y_prediction)
        print('cost \n', sess.run(cost, feed_dict={x_: XOR_X, y_: XOR_Y}))
    # Si hemos conseguido nuestro objetivo (en este caso que el error sea muy pequeño) dejamos de entrenar.
    # Esta es la cota que hemos puesto en nuestro aprendizaje. La cota puede ser mayor o menor según nuestros intereses
    # y según nuestro problema.
    if coste < 0.05:
        # Comprobamos que es correcto
        feed = {x_: [[0, 0], [0, 0], [0, 0], [1, 1]], y_: [[0], [0], [0], [0]]}
        y_prediction_final = y2.eval(feed_dict=feed)
        print("y_final: ", y_prediction_final)
        break
t_end = time.clock()
print('Elapsed time ', t_end - t_start)
plt.plot(y0_, label='y0_')
plt.plot(y1_, label='y1_')
plt.plot(y2_, label='y2_')
plt.plot(y3_, label='y3_')
plt.legend()
plt.show()
plt.plot(costs, label='costs')
plt.legend()
plt.show()