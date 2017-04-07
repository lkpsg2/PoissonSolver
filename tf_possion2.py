# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from func import *

with tf.name_scope('input_data') as scope:
    x = tf.placeholder("float",shape=[None,66,66,1],name='input')
    y_actual = tf.placeholder("float",shape=[None,32,32,1],name='results')
    keep_prob = tf.placeholder("float",name='drop_out')

    tempx=x
    tempy=tf.abs(tf.div(tf.log(y_actual),tf.log(10.0)))

def model(is_training):
    #set up the network
    '''layer1'''
    ten =10 * tf.ones([40,32,32,1])
    with tf.name_scope('layer1') as scope:
        W_conv1 = weight_variable([3,3,1,8])
        b_conv1 = bias_variable([8])
        h_conv1 = conv2d_V(tempx,W_conv1) + b_conv1
        l1 = tf.nn.relu(h_conv1)

        h_pool1 = max_pool(l1)

    '''layer2'''
    tf.summary.histogram("/weights",W_conv1)
    with tf.name_scope('layer2') as scope:
        W_conv2 = weight_variable([3,3,8,8])
        b_conv2 = bias_variable([8])
        h_conv2 = conv2d_S(l1,W_conv2) + b_conv2
        l2 = tf.nn.relu(h_conv2)

        h_pool2_l2 = max_pool(l2)

        W_pool2 = weight_variable([3,3,8,8])
        b_pool2 = bias_variable([8])
        h_pool2 = conv2d_S(h_pool1,W_pool2) + b_pool2
        l_pool2 = tf.nn.relu(h_pool2)

#        w1 = weight_variable([2,2,8,8])
#        y_dconv = tf.nn.conv2d_transpose(l_pool2,w1,[40,32,32,8],[1,2,2,1],'VALID')
        y_layer2=tf.add(l_pool2,h_pool2_l2)

    '''layer3'''
    with tf.name_scope('layer3') as scope:
        W_conv3 = weight_variable([5,5,8,8])
        b_conv3 = bias_variable([8])
        h_conv3 = conv2d_S(y_layer2,W_conv3) + b_conv3
        l3 = tf.nn.relu(h_conv3)

    '''layer4'''
    with tf.name_scope('layer4') as scope:
        W_conv4 = weight_variable([5,5,8,8])
        b_conv4 = bias_variable([8])
        h_conv4 = conv2d_S(l3,W_conv4) + b_conv4
        l4 = tf.nn.relu(h_conv4)

    W_conv5 = weight_variable([1,1,8,8])
    b_conv5 = bias_variable([8])
    l5 = conv2d_S(l4,W_conv5) + b_conv5

    W_conv6 = weight_variable([1,1,8,1])
    b_conv6 = bias_variable([1])
    y_predict = conv2d_S(l5,W_conv6) + b_conv6

    with tf.name_scope('eval_error'):
        with tf.name_scope('rmse') as scope:
            rmse = tf.sqrt(tf.reduce_mean(tf.div(tf.reduce_mean(tf.square(tempy - y_predict),[1,2]),tf.reduce_mean(tf.square(tempy),[1,2]))))
        with tf.name_scope('db') as scope:
            #db = 20*tf.log(tf.add(tf.div(tf.abs(tempy-y_predict),tf.abs(tempy)),1e-10),10)
            db = 20*tf.div(tf.log(tf.add(tf.div(tf.abs(tf.pow(ten,tempy)-tf.pow(ten,y_predict)),tf.abs(tf.pow(ten,tempy))),1e-10)),tf.log(10.0))
            mean_db = tf.reduce_mean(db)
    db_summary = tf.summary.histogram('db',db)
    return y_predict,rmse,mean_db
