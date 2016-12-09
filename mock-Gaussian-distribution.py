import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mu, sigma = -1,01
xs = np.linspace(-5, 5 ,1000)
#plt.plot(xs, norm.pdf(xs, loc=mu, scale=sigma))

TRAIN_ITERS=10000
BAT_SIZE=200
M = 200
def mlp(input, output_dim):
    w1=tf.get_variable("w0", [input.get_shape()[1], 6], initializer=tf.random_normal_initializer())
    b1=tf.get_variable("b0", [6], initializer=tf.constant_initializer(0.0))
    w2=tf.get_variable("w1", [6, 5], initializer=tf.random_normal_initializer())
    b2=tf.get_variable("b1", [5], initializer=tf.constant_initializer(0.0))
    w3=tf.get_variable("w2", [5,output_dim], initializer=tf.random_normal_initializer())
    b3=tf.get_variable("b2", [output_dim], initializer=tf.constant_initializer(0.0))
    fc1=tf.nn.tanh(tf.matmul(input,w1)+b1)
    fc2=tf.nn.tanh(tf.matmul(fc1,w2)+b2)
    fc3=tf.nn.tanh(tf.matmul(fc2,w3)+b3)
    return fc3, [w1,b1,w2,b2,w3,b3]  

def momentum_optimizer(loss, var_list):
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
            0.001,
            batch,
            TRAIN_ITERS // 4,

            0.96,
            staircase=True)
    optimizer=tf.train.MomentumOptimizer(learning_rate, 0.6).minimize(loss, global_step=batch, 
            var_list=var_list)
    return optimizer

with tf.variable_scope("D_pre"):
    input_node = tf.placeholder(tf.float32, shape=(BAT_SIZE, 1))
    train_labels = tf.placeholder(tf.float32, shape=(BAT_SIZE, 1))
    D, theta = mlp(input_node, 1)
    loss = tf.reduce_mean(tf.square(D-train_labels))

optimizer = momentum_optimizer(loss, None)
sess=tf.InteractiveSession()
tf.initialize_all_variables().run()

def plot_d0(D, input_node):
    f, ax = plt.subplots(1)
    #plot data
    xs = np.linspace(-5, 5, 1000)
    ax.plot(xs, norm.pdf(xs, loc=mu, scale=sigma), label='p_data')
    # decision boundary
    xs = np.linspace(-5, 5, 1000)
    ds = np.zeros((1000, 1)) # decision surface
    
    for i in range(1000 / BAT_SIZE):
        x=np.reshape(xs[BAT_SIZE*i:BAT_SIZE*(i+1)], (BAT_SIZE, 1))
        ds[BAT_SIZE*i:BAT_SIZE*(i+1)] = sess.run(D, {input_node: x})
    ax.plot(xs, ds, label="decision boundary")
    ax.set_ylim(0, 1.1)
    plt.legend()
    plt.show()


plot_d0(D,input_node)
plt.title('Initial Decision Boundary')

lh=np.zeros(1000)
for i in range(1000):
    d=(np.random.random(M)-0.5) * 10.0 # instead of sampling only from gaussian, want the domain to be covered as uniformly as possible
    labels=norm.pdf(d,loc=mu,scale=sigma)
    lh[i],_=sess.run([loss,optimizer], {input_node: np.reshape(d,(M,1)), train_labels: np.reshape(labels,(M,1))})

plt.plot(lh)
plt.title('Training loss')

plot_d0(D, input_node)

weightsD = sess.run(theta)
sess.close()


with tf.variable_scope("G"):
    z_node=tf.placeholder(tf.float32, shape=(M,1)) # M uniform01 floats
    G,theta_g=mlp(z_node,1) # generate normal transformation of Z
    G=tf.mul(5.0,G) # scale up by 5 to match range

with tf.variable_scope("D") as scope:
        # D(x)
        x_node=tf.placeholder(tf.float32, shape=(M,1)) # input M normally distributed floats
        fc,theta_d=mlp(x_node,1) # output likelihood of being normally distributed
        D1=tf.maximum(tf.minimum(fc,.99), 0.01) # clamp as a probability
        # make a copy of D that uses the same variables, but takes in G as input
        scope.reuse_variables()
        fc,theta_d=mlp(G,1)
        D2=tf.maximum(tf.minimum(fc,.99), 0.01)


