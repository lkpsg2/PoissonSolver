# -*- coding: utf-8 -*-
import tensorflow as tf
import scipy.io as sio
import numpy as np
import random
import matplotlib.pyplot as plt
import input_data
import tf_possion

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.0001
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,40000, 0.8, staircase=True)

sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
#with tf.device('/gpu:2'):
y_predict,rmse,grad_x_rmse,grad_y_rmse,mean_db,f_obj = tf_possion.model()
#y_predict_test,rmse_test,mean_db_test = tf_possion.model(is_training=False)
#with tf.device('/gpu:1'):
train_step = tf.train.AdamOptimizer(learning_rate).minimize(f_obj,global_step=global_step)
#train_step = tf.train.GradientDescentOptimizer(0.001).minimize(rmse)

merged_summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('./logs', sess.graph)
#tf.reset_default_graph()

sess.run(tf.global_variables_initializer())
#sess.run(tf_possion.op)
x_train,y_train = input_data.input_data(test=False)
x_test,y_test = input_data.input_data(test=True)

epochs = 10000
train_size = x_train.shape[0]
#print(train_size)
global batch
batch = 40
test_size = x_test.shape[0]

train_index = list(range(x_train.shape[0]))
test_index = list(range(x_test.shape[0]))
for i in range(epochs):

    random.shuffle(train_index)
    random.shuffle(test_index)
    x_train,y_train = x_train[train_index],y_train[train_index]
    x_test,y_test = x_test[test_index],y_test[test_index]

    for j in range(0,train_size,batch):
        train_step.run(feed_dict={tf_possion.x:x_train[j:j+batch],tf_possion.y_actual:y_train[j:j+batch],tf_possion.keep_prob:0.5})
    

    temp = 0
    train_loss=0
#        if (i%30)==0:
    for j in range(0,train_size,batch):
        train_loss = rmse.eval(feed_dict={tf_possion.x:x_train[j:j+batch],tf_possion.y_actual:y_train[j:j+batch],tf_possion.keep_prob: 1.0})
        temp = temp + train_loss
    train_loss = temp/(train_size/batch)

    temp_loss = 0
    temp_grad_x = 0
    temp_db = 0
    temp_fobj = 0
    gradx_result = 0
    fobj = 0
    meandb = 0
    loss = 0
#        if (i%30)==0:
    for j in range(0,test_size,batch):
        loss = rmse.eval(feed_dict={tf_possion.x:x_test[j:j+batch],tf_possion.y_actual:y_test[j:j+batch],tf_possion.keep_prob: 1.0})
        meandb = mean_db.eval(feed_dict={tf_possion.x:x_test[j:j+batch],tf_possion.y_actual:y_test[j:j+batch],tf_possion.keep_prob: 1.0})
        fobj = f_obj.eval(feed_dict={tf_possion.x:x_test[j:j+batch],tf_possion.y_actual:y_test[j:j+batch],tf_possion.keep_prob: 1.0})
        gradx_result = grad_x_rmse.eval(feed_dict={tf_possion.x:x_test[j:j+batch],tf_possion.y_actual:y_test[j:j+batch],tf_possion.keep_prob: 1.0})
        temp_loss = temp_loss+loss
        temp_db = temp_db+meandb
        temp_fobj = temp_fobj+fobj
        temp_grad_x = temp_grad_x + gradx_result

    loss = temp_loss/(test_size/batch)
    meandb = temp_db/(test_size/batch)
    fobj = temp_fobj/(test_size/batch)
    gradx_result = temp_grad_x/(test_size/batch)
    if i==750:
        y_print = y_predict.eval(feed_dict={tf_possion.x:x_test[0:batch],tf_possion.y_actual:y_test[0:batch],tf_possion.keep_prob: 1.0})

        #result = tf.scalar_summary('y_result',y_print)
        #print 'y_print {0}'.format(y_print)
        sio.savemat('result.mat',{'aa':y_test[0:batch],'aa_test':y_print})

    summary_str = sess.run(merged_summary_op,feed_dict={tf_possion.x:x_test[0:batch],tf_possion.y_actual:y_test[0:batch],tf_possion.keep_prob: 1.0})
    summary_writer.add_summary(summary_str,j)
    print ('epoch {0} done! train_loss:{1} test_loss:{2} grad_x:{3} f_obj:{4} db:{5} global_step:{6} learning rate:{7}'.format(i,train_loss, loss,gradx_result,fobj,meandb,global_step.eval(),learning_rate.eval()))