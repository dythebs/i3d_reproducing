import inception_v1_3d
import ckpt_util
import input_data
import numpy as np 
import os
import tensorflow.contrib.slim as slim
import tensorflow as tf 


'''
建立计算图
从ckpt文件中恢复参数
创建数据集
'''
keep_prob = tf.placeholder(dtype=tf.float32)
with tf.variable_scope('rgb'):
	rgb_input = tf.placeholder(dtype=tf.uint8, shape=[None, 64, 224, 224, 3])
	rgb_input_process = tf.cast(rgb_input, tf.float32) / 128. -1
	rgb_logits = inception_v1_3d.inception_v1_3d(rgb_input_process, keep_prob, 101)
with tf.variable_scope('flow'):
	flow_input = tf.placeholder(dtype=tf.float32, shape=[None, 64, 224, 224, 2])
	flow_input_process = flow_input / 20.
	flow_logits = inception_v1_3d.inception_v1_3d(flow_input_process, keep_prob, 101)

logits = rgb_logits + flow_logits

y = tf.nn.softmax(logits)
y_ = tf.placeholder(dtype=tf.float32,shape=[None, 101])

cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y, 1e-7, 1)))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
ckpt_util.restore(sess)
one_element = input_data.read_data(sess)
for i in range(3000):
	element = sess.run(one_element)
	rgb_datas = element[0]
	flow_datas = element[1]
	labels = input_data.one_hot([b.decode() for b in element[-1].tolist()])
	sess.run(train_step, feed_dict={rgb_input:rgb_datas, flow_input:flow_datas, y_:labels, keep_prob:0.6})
	if i % 50 == 0:
		element = sess.run(one_element)
		rgb_datas = element[0]
		flow_datas = element[1]
		labels = input_data.one_hot([b.decode() for b in element[-1].tolist()])
		train_accuracy, loss = sess.run([accuracy, cross_entropy] ,
			feed_dict={rgb_input:rgb_datas, flow_input:flow_datas, y_:labels, keep_prob:1})
		print("step %d, train accuracy %g, loss %g" % (i, train_accuracy, loss))