import tensorflow as tf

#function to initial the weight W
def weight_variable(shape,name):
#    initial = tf.truncated_normal(shape, stddev=0.1)
    initial = tf.get_variable(name,shape,initializer=tf.contrib.layers.xavier_initializer_conv2d()) 
    return initial

#function to initial the bias b
def bias_variable(shape):
    #initial = tf.constant(0.1,shape=shape)
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def zeros_variable(shape):
    initial = tf.zeros(shape)
    return tf.Variable(initial)

#function to set up conv layer
def conv2d_S(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def conv2d_V(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='VALID')

#function to set up pooling layer
def max_pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def norm(featuremaps):
    return tf.nn.lrn(featuremaps, 4, bias=1, alpha=0.0001, beta=0.75)

def batch_norm(inputs, is_training, decay = 0.999):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        train_mean = tf.assign(pop_mean,pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,batch_mean, batch_var, beta, scale, 0.001)
    else:
        return tf.nn.batch_normalization(inputs,pop_mean, pop_var, beta, scale, 0.001)

def grad(x):
###
# x is y_predict by batch *32*32*1
###

    trans_x = tf.transpose(x,[2,1,0,3])
    trans_y = tf.transpose(x,[1,0,2,3])
#    gradx = zeros_variable([32,32,40,1])
    gradx = zeros_variable([32,32,40,1]) 
    grady = zeros_variable([32,40,32,1])
    sess=tf.Session()
#    sess.run(tf.global_variables_initializer())

    op=gradx[0].assign(trans_x[1] - trans_x[0])
    tf_train.sess.run(op)
    op=gradx[31].assign(trans_x[31] - trans_x[30])
    sess.run(op)
    op=grady[0].assign(trans_y[1] - trans_y[0])
    sess.run(op)
    op=grady[31].assign(trans_y[31] - trans_y[30])
    sess.run(op)
    for i in range(1,31):
        op=gradx[i].assign(0.5 * (trans_x[i+1] - trans_x[i-1]))
        sess.run(op)
        op=grady[i].assign(0.5 * (trans_y[i+1] - trans_y[i-1]))
        sess.run(op)

    gradx = tf.transpose(gradx,[2,1,0,3])
    grady = tf.transpose(grady,[1,0,2,3])
    return gradx,grady