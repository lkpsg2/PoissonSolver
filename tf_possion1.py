# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
#function to initial the weight W
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#function to initial the bias b
def bias_variable(shape):
    #initial = tf.constant(0.1,shape=shape)
    initial = tf.truncated_normal(shape, stddev=0.1)
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

with tf.name_scope('input_data') as scope:
    x = tf.placeholder("float",shape=[None,66,66,1],name='input')
    y_actual = tf.placeholder("float",shape=[None,64,64,1],name='results')
    keep_prob = tf.placeholder("float",name='drop_out')
#    x = norm(x)
#    y_actual = norm(y_actual)

def model(is_training):
    #set up the network
    with tf.name_scope('layer1') as scope:
        with tf.name_scope('W_conv1') as scope:
            W_conv1 = weight_variable([3,3,1,16])
        with tf.name_scope('b_conv1') as scope:
            b_conv1 = bias_variable([16])

        h_conv1 = conv2d_V(x,W_conv1) + b_conv1
        bn1 = batch_norm(h_conv1, is_training)
        with tf.name_scope('l1') as scope:
            l1 = tf.nn.relu(bn1)
        with tf.name_scope('h_pool1') as scope: 
            h_pool1 = max_pool(l1)
        tf.summary.histogram("/weights",W_conv1)
    with tf.name_scope('layer2') as scope:
        with tf.name_scope('W_conv2') as scope:
            W_conv2 = weight_variable([3,3,16,16])
        with tf.name_scope('b_conv2') as scope:
            b_conv2 = bias_variable([16])

        h_conv2 = conv2d_S(h_pool1,W_conv2) + b_conv2
        bn2 = batch_norm(h_conv2, is_training)
        with tf.name_scope('l2') as scope:
            l2 = tf.nn.relu(bn2)
        with tf.name_scope('h_pool2') as scope:
            h_pool2 = max_pool(l2)
    with tf.name_scope('layer3') as scope:
        with tf.name_scope('W_conv3') as scope:
            W_conv3 = weight_variable([3,3,16,16])
        with tf.name_scope('b_conv3') as scope:
            b_conv3 = bias_variable([16])

        h_conv3 = conv2d_S(h_pool2,W_conv3) + b_conv3
        bn3 = batch_norm(h_conv3, is_training)
        with tf.name_scope('l3') as scope:
            l3 = tf.nn.relu(bn3)

    W_conv4 = weight_variable([3,3,16,16])
    b_conv4 = bias_variable([16])
    h_conv4 = conv2d_S(l3,W_conv4) + b_conv4
    bn4 = batch_norm(h_conv4, is_training)
    l4 = tf.nn.relu(bn4)
    '''
    with tf.name_scope('full_connect') as scope:
        with tf.name_scope('W_fc1') as scope:
            W_fc1 = weight_variable([3 * 3 * 128,1024])
        with tf.name_scope('b_fc1') as scope:
            b_fc1 = bias_variable([1024])
        with tf.name_scope('h_pool3_flat') as scope:
            h_pool3_flat = tf.reshape(h_pool3,[-1,3 * 3 * 128])
        with tf.name_scope('y_predict') as scope:
            h_fc1 = tf.matmul(h_pool3_flat,W_fc1) + b_fc1

    with tf.name_scope('dropout') as scope:
        h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
   '''
    W_conv5 = weight_variable([1,1,16,32])
    b_conv5 = bias_variable([32])
    h_conv5 = conv2d_S(l4,W_conv5) + b_conv5
    bn5 = batch_norm(h_conv5, is_training)
    l5 = tf.nn.relu(bn5)

    W_conv6 = weight_variable([1,1,32,32])
    b_conv6 = bias_variable([32])
    h_conv6 = conv2d_S(l5,W_conv6) + b_conv6
    bn6 = batch_norm(h_conv6, is_training)
    l6 = tf.nn.relu(bn6)
    '''
    with tf.name_scope('y_predict') as scope:
        with tf.name_scope('W_fc2') as scope:
            W_fc2 = weight_variable([16 * 16 * 32,500])
        with tf.name_scope('b_fc2') as scope:
            b_fc2 = bias_variable([500])
        with tf.name_scope('h_pool3_flat') as scope:
            h_fc2 = tf.reshape(norm6,[-1,16 * 16 * 32])

        with tf.name_scope('y_predict') as scope:
            y_fc2 = tf.nn.relu(tf.matmul(h_fc2,W_fc2) + b_fc2)

            W_fc3 = weight_variable([500,28])
            b_fc3 = bias_variable([28])
            y_predict = tf.matmul(y_fc2,W_fc3) + b_fc3
    '''

    w1 = weight_variable([2,2,1,32])
    y_dconv = tf.nn.conv2d_transpose(l6,w1,[40,32,32,1],[1,2,2,1],'VALID')
#    y_dconv_norm = norm(y_dconv)

    w2 = weight_variable([2,2,1,1])
    y_predict = tf.nn.conv2d_transpose(y_dconv,w2,[40,64,64,1],[1,2,2,1],'VALID')

    with tf.name_scope('eval_error'):
        with tf.name_scope('rmse') as scope:
            rmse = tf.sqrt(tf.reduce_mean(tf.div(tf.reduce_mean(tf.square(y_actual - y_predict),[1,2]),tf.reduce_mean(tf.square(y_actual),[1,2]))))
        with tf.name_scope('db') as scope:
            db = 20*tf.div(tf.log(tf.add(tf.div(tf.abs(y_actual-y_predict),tf.abs(y_actual)),1e-10)),tf.log(10.0))
            #db = 20*tf.log(tf.add(tf.div(tf.abs(y_actual-y_predict),tf.abs(y_actual)),1e-10))
            mean_db = tf.reduce_mean(db)
    db_summary = tf.summary.histogram('db',db)
    return y_predict,rmse,mean_db
