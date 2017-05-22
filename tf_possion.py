# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from func import *

with tf.name_scope('input_data') as scope:
    x = tf.placeholder("float",shape=[None,64,64,1],name='input')
    d = tf.placeholder("float",shape=[None,64,64,1],name='input')
    y_actual = tf.placeholder("float",shape=[None,32,32,1],name='results')
    keep_prob = tf.placeholder("float",name='drop_out')

    log10_y=tf.abs(tf.div(tf.log(y_actual),tf.log(10.0)))

def model():
    #set up the network
    '''layer1'''
    ten =10 * tf.ones([40,32,32,1])

    with tf.name_scope('layer_d1') as scope:
        W_conv_d1 = weight_variable([3,3,1,8],name='w_conv_d1')
        b_conv_d1 = bias_variable([8])
        h_conv_d1 = conv2d_S(d,W_conv_d1) + b_conv_d1
        l_d1 = tf.nn.relu(h_conv_d1)

    with tf.name_scope('layer_d2') as scope:
        W_conv_d2 = weight_variable([3,3,8,1],name='w_conv_d2')
        b_conv_d2 = bias_variable([1])
        h_conv_d2 = conv2d_S(l_d1,W_conv_d2) + b_conv_d2
        l_d2 = tf.nn.relu(h_conv_d2)

    tempx=tf.concat(axis=3,values=[d,x])

    with tf.name_scope('layer1') as scope:
        W_conv1 = weight_variable([11,11,2,16],name='w_conv1')
        b_conv1 = bias_variable([16])
        h_conv1 = conv2d_V(tempx,W_conv1) + b_conv1
        l1 = tf.nn.relu(h_conv1)

    '''layer2'''
    tf.summary.histogram("/weights_layer1",W_conv1)
    with tf.name_scope('layer2') as scope:
        W_conv2 = weight_variable([11,11,16,32],name='w_conv2')
        b_conv2 = bias_variable([32])
        h_conv2 = conv2d_V(l1,W_conv2) + b_conv2
        l2 = tf.nn.relu(h_conv2)


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
        W_conv5 = weight_variable([3,3,64,64],name='w_conv5')
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

#        w1 = weight_variable([2,2,8,8])
#        y_dconv = tf.nn.conv2d_transpose(l_pool2,w1,[40,32,32,8],[1,2,2,1],'VALID')

    with tf.name_scope('eval_error'):
        with tf.name_scope('rmse') as scope:
            #grad_x_temp,grad_y_temp=grad(y_predict)

            trans_x = tf.transpose(y_predict,[2,1,0,3])
            trans_y = tf.transpose(y_predict,[1,0,2,3])

            pack_x = []
            pack_y = []
            pack_x = tf.stack([trans_x[1] - trans_x[0]])
            pack_y = tf.stack([trans_y[1] - trans_y[0]])

            for i in range(1,31):
                pack_x = tf.concat( axis = 0 , values = [ pack_x , [ 0.5 * (trans_x[i+1] - trans_x[i-1]) ] ] )
                pack_y = tf.concat( axis = 0 , values = [ pack_y , [ 0.5 * (trans_y[i+1] - trans_y[i-1]) ] ] )

            pack_x = tf.concat( axis = 0 , values = [ pack_x , [ trans_x [ 31 ] - trans_x [ 30 ] ] ] )
            pack_y = tf.concat( axis = 0 , values = [ pack_y , [ trans_y [ 31 ] - trans_y [ 30 ] ] ] )
            gradx = tf.transpose(pack_x , [2,1,0,3])
            grady = tf.transpose(pack_y , [1,0,2,3])

#            rmse = tf.sqrt(tf.reduce_mean(tf.div(tf.reduce_mean(tf.square(tempy - y_predict),[1,2]),tf.reduce_mean(tf.square(tempy),[1,2]))))
            rmse = tf.reduce_mean(tf.reduce_mean(tf.div(tf.square(log10_y - y_predict),tf.square(log10_y)),[1,2]))
#            grad_x_rmse = tf.sqrt(tf.reduce_mean(tf.div(tf.reduce_mean(tf.square(gradx),[1,2]),tf.reduce_mean(tf.square(tempy),[1,2]))))
            grad_x_rmse = tf.reduce_mean(tf.reduce_mean(tf.div(tf.square(gradx),tf.square(log10_y)),[1,2]))
#            grad_y_rmse = tf.sqrt(tf.reduce_mean(tf.div(tf.reduce_mean(tf.square(grady),[1,2]),tf.reduce_mean(tf.square(tempy),[1,2]))))
            grad_y_rmse = tf.reduce_mean(tf.reduce_mean(tf.div(tf.square(grady),tf.square(log10_y)),[1,2]))
#            f_obj = 1 * rmse+0* (tf.sqrt(tf.square(grad_x_rmse) + tf.square(grad_y_rmse)))
            f_obj = rmse + 0 *(grad_x_rmse + grad_y_rmse)
        with tf.name_scope('db') as scope:
            #db = 20*tf.log(tf.add(tf.div(tf.abs(tempy-y_predict),tf.abs(tempy)),1e-10),10)
            db = 20*tf.div(tf.log(tf.add(tf.div(tf.abs(tf.pow(ten,log10_y)-tf.pow(ten,y_predict)),tf.abs(tf.pow(ten,log10_y))),1e-10)),tf.log(10.0))
            mean_db = tf.reduce_mean(db)
    db_summary = tf.summary.histogram('db',mean_db)
    return y_predict,rmse,grad_x_rmse,grad_y_rmse,mean_db,f_obj,l_d2
