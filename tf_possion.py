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
        W_conv1 = weight_variable([11,11,1,16],name='w_conv1')
        b_conv1 = bias_variable([16])
        h_conv1 = conv2d_V(tempx,W_conv1) + b_conv1
        l1 = tf.nn.relu(h_conv1)

    '''layer2'''
    tf.summary.histogram("/weights",W_conv1)
    with tf.name_scope('layer2') as scope:
        W_conv2 = weight_variable([11,11,16,32],name='w_conv2')
        b_conv2 = bias_variable([32])
        h_conv2 = conv2d_V(l1,W_conv2) + b_conv2
        l2 = tf.nn.relu(h_conv2)

#        w1 = weight_variable([2,2,8,8])
#        y_dconv = tf.nn.conv2d_transpose(l_pool2,w1,[40,32,32,8],[1,2,2,1],'VALID')

    '''layer3'''
    with tf.name_scope('layer3') as scope:
        W_conv3 = weight_variable([5,5,32,64],name='w_conv3')
        b_conv3 = bias_variable([64])
        h_conv3 = conv2d_V(l2,W_conv3) + b_conv3
        l3 = tf.nn.relu(h_conv3)

    '''layer4'''
    with tf.name_scope('layer4') as scope:
        W_conv4 = weight_variable([5,5,64,64],name='w_conv4')
        b_conv4 = bias_variable([64])
        h_conv4 = conv2d_V(l3,W_conv4) + b_conv4
        l4 = tf.nn.relu(h_conv4)

    with tf.name_scope('layer5') as scope:
        W_conv5 = weight_variable([5,5,64,64],name='w_conv5')
        b_conv5 = bias_variable([64])
        h_conv5 = conv2d_V(l4,W_conv5) + b_conv5
        l5 = tf.nn.relu(h_conv5)

    with tf.name_scope('layer6') as scope:
        W_conv6 = weight_variable([3,3,64,128],name='w_conv6')
        b_conv6 = bias_variable([128])
        h_conv6 = conv2d_V(l5,W_conv6) + b_conv6
        l6 = tf.nn.relu(h_conv6)

    with tf.name_scope('layer7') as scope:
        W_conv7 = weight_variable([1,1,128,64],name='w_conv7')
        b_conv7 = bias_variable([64])
        h_conv7 = conv2d_V(l6,W_conv7) + b_conv7
        l7 = tf.nn.relu(h_conv7)

    with tf.name_scope('layer8') as scope:
        W_conv8 = weight_variable([1,1,64,1],name='w_conv8')
        b_conv8 = bias_variable([1])
        y_predict = conv2d_V(l7,W_conv8) + b_conv8

    with tf.name_scope('eval_error'):
        with tf.name_scope('rmse') as scope:
            rmse = tf.sqrt(tf.reduce_mean(tf.div(tf.reduce_mean(tf.square(tempy - y_predict),[1,2]),tf.reduce_mean(tf.square(tempy),[1,2]))))
#            grad_x = tf.sqrt(tf.reduce_mean(tf.div(tf.reduce_mean(tf.square(tf.gradients(y_predict,y_predict)),[1,2]),tf.reduce_mean(tf.square(tempy),[1,2])))) 
            y_predict_trans = tf.transpose(y_predict)
#            grad_y = tf.sqrt(tf.reduce_mean(tf.div(tf.reduce_mean(tf.square(tf.gradients(y_predict_trans,y_predict_trans)),[1,2]),tf.reduce_mean(tf.square(tempy),[1,2])))) 
#            f_obj = 1 * rmse + 0 *grad_x + 0 * grad_y
            f_obj = rmse
 
        with tf.name_scope('db') as scope:
            #db = 20*tf.log(tf.add(tf.div(tf.abs(tempy-y_predict),tf.abs(tempy)),1e-10),10)
            db = 20*tf.div(tf.log(tf.add(tf.div(tf.abs(tf.pow(ten,tempy)-tf.pow(ten,y_predict)),tf.abs(tf.pow(ten,tempy))),1e-10)),tf.log(10.0))
            mean_db = tf.reduce_mean(db)
    db_summary = tf.summary.histogram('db',db)
    return y_predict,rmse,mean_db,f_obj
