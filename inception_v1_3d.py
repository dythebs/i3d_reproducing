import tensorflow.contrib.slim as slim
import tensorflow as tf 

def inception_v1_3d(inputs, keep_prob, num_classes):
	with tf.variable_scope('InceptionV1_3d'):
		with slim.arg_scope([slim.conv3d, slim.fully_connected],
			weights_initializer=tf.truncated_normal_initializer(stddev=0.001)):
			with slim.arg_scope([slim.conv3d, slim.max_pool3d],
				stride=1, padding='SAME'):
				with slim.arg_scope([slim.conv3d],
					normalizer_fn=slim.batch_norm):
					with slim.arg_scope([slim.batch_norm, slim.dropout],
						is_training=True):

						net = slim.conv3d(inputs, 64, [7, 7, 7], stride=2, scope='Conv2d_1a_7x7')
						net = slim.max_pool3d(net, [1, 3, 3], stride=[1, 2, 2], scope='MaxPool_2a_3x3')
						net = slim.conv3d(net, 64, [1, 1, 1], scope='Conv2d_2b_1x1')
						net = slim.conv3d(net, 192, [3, 3, 3], scope='Conv2d_2c_3x3')
						net = slim.max_pool3d(net, [1, 3, 3], stride=[1, 2, 2], scope='MaxPool_3a_3x3')

						with tf.variable_scope('Mixed_3b'):
							with tf.variable_scope('Branch_0'):
								branch_0 = slim.conv3d(net, 64, [1, 1, 1], scope='Conv2d_0a_1x1')
							with tf.variable_scope('Branch_1'):
								branch_1 = slim.conv3d(net, 96, [1, 1, 1], scope='Conv2d_0a_1x1')
								branch_1 = slim.conv3d(branch_1, 128, [3, 3, 3], scope='Conv2d_0b_3x3')
							with tf.variable_scope('Branch_2'):
								branch_2 = slim.conv3d(net, 16, [1, 1, 1], scope='Conv2d_0a_1x1')
								branch_2 = slim.conv3d(branch_2, 32, [3, 3, 3], scope='Conv2d_0b_3x3')
							with tf.variable_scope('Branch_3'):
								branch_3 = slim.max_pool3d(net, [3, 3, 3], scope='MaxPool_0a_3x3')
								branch_3 = slim.conv3d(branch_3, 32, [1, 1, 1], scope='Conv2d_0b_1x1')
							net = tf.concat(axis=4, values=[branch_0, branch_1, branch_2, branch_3])

						with tf.variable_scope('Mixed_3c'):
							with tf.variable_scope('Branch_0'):
			 					branch_0 = slim.conv3d(net, 128, [1, 1, 1], scope='Conv2d_0a_1x1')
							with tf.variable_scope('Branch_1'):
								branch_1 = slim.conv3d(net, 128, [1, 1, 1], scope='Conv2d_0a_1x1')
								branch_1 = slim.conv3d(branch_1, 192, [3, 3, 3], scope='Conv2d_0b_3x3')
							with tf.variable_scope('Branch_2'):
								branch_2 = slim.conv3d(net, 32, [1, 1, 1], scope='Conv2d_0a_1x1')
								branch_2 = slim.conv3d(branch_2, 96, [3, 3, 3], scope='Conv2d_0b_3x3')
							with tf.variable_scope('Branch_3'):
								branch_3 = slim.max_pool3d(net, [3, 3, 3], scope='MaxPool_0a_3x3')
								branch_3 = slim.conv3d(branch_3, 64, [1, 1, 1], scope='Conv2d_0b_1x1')
							net = tf.concat(axis=4, values=[branch_0, branch_1, branch_2, branch_3])

						net = slim.max_pool3d(net, [3, 3, 3], stride=2, scope='MaxPool_4a_3x3')

						with tf.variable_scope('Mixed_4b'):
							with tf.variable_scope('Branch_0'):
								branch_0 = slim.conv3d(net, 192, [1, 1, 1], scope='Conv2d_0a_1x1')
							with tf.variable_scope('Branch_1'):
								branch_1 = slim.conv3d(net, 96, [1, 1, 1], scope='Conv2d_0a_1x1')
								branch_1 = slim.conv3d(branch_1, 208, [3, 3, 3], scope='Conv2d_0b_3x3')
							with tf.variable_scope('Branch_2'):
								branch_2 = slim.conv3d(net, 16, [1, 1, 1], scope='Conv2d_0a_1x1')
								branch_2 = slim.conv3d(branch_2, 48, [3, 3, 3], scope='Conv2d_0b_3x3')
							with tf.variable_scope('Branch_3'):
								branch_3 = slim.max_pool3d(net, [3, 3, 3], scope='MaxPool_0a_3x3')
								branch_3 = slim.conv3d(branch_3, 64, [1, 1, 1], scope='Conv2d_0b_1x1')
							net = tf.concat(axis=4, values=[branch_0, branch_1, branch_2, branch_3])

						with tf.variable_scope('Mixed_4c'):
							with tf.variable_scope('Branch_0'):
								branch_0 = slim.conv3d(net, 160, [1, 1, 1], scope='Conv2d_0a_1x1')
							with tf.variable_scope('Branch_1'):
								branch_1 = slim.conv3d(net, 112, [1, 1, 1], scope='Conv2d_0a_1x1')
								branch_1 = slim.conv3d(branch_1, 224, [3, 3, 3], scope='Conv2d_0b_3x3')
							with tf.variable_scope('Branch_2'):
								branch_2 = slim.conv3d(net, 24, [1, 1, 1], scope='Conv2d_0a_1x1')
								branch_2 = slim.conv3d(branch_2, 64, [3, 3, 3], scope='Conv2d_0b_3x3')
							with tf.variable_scope('Branch_3'):
								branch_3 = slim.max_pool3d(net, [3, 3, 3], scope='MaxPool_0a_3x3')
								branch_3 = slim.conv3d(branch_3, 64, [1, 1, 1], scope='Conv2d_0b_1x1')
							net = tf.concat(axis=4, values=[branch_0, branch_1, branch_2, branch_3])

						with tf.variable_scope('Mixed_4d'):
							with tf.variable_scope('Branch_0'):
								branch_0 = slim.conv3d(net, 128, [1, 1, 1], scope='Conv2d_0a_1x1')
							with tf.variable_scope('Branch_1'):
								branch_1 = slim.conv3d(net, 128, [1, 1, 1], scope='Conv2d_0a_1x1')
								branch_1 = slim.conv3d(branch_1, 256, [3, 3, 3], scope='Conv2d_0b_3x3')
							with tf.variable_scope('Branch_2'):
								branch_2 = slim.conv3d(net, 24, [1, 1, 1], scope='Conv2d_0a_1x1')
								branch_2 = slim.conv3d(branch_2, 64, [3, 3, 3], scope='Conv2d_0b_3x3')
							with tf.variable_scope('Branch_3'):
								branch_3 = slim.max_pool3d(net, [3, 3, 3], scope='MaxPool_0a_3x3')
								branch_3 = slim.conv3d(branch_3, 64, [1, 1, 1], scope='Conv2d_0b_1x1')
							net = tf.concat(axis=4, values=[branch_0, branch_1, branch_2, branch_3])

						with tf.variable_scope('Mixed_4e'):
							with tf.variable_scope('Branch_0'):
								branch_0 = slim.conv3d(net, 112, [1, 1, 1], scope='Conv2d_0a_1x1')
							with tf.variable_scope('Branch_1'):
								branch_1 = slim.conv3d(net, 144, [1, 1, 1], scope='Conv2d_0a_1x1')
								branch_1 = slim.conv3d(branch_1, 288, [3, 3, 3], scope='Conv2d_0b_3x3')
							with tf.variable_scope('Branch_2'):
								branch_2 = slim.conv3d(net, 32, [1, 1, 1], scope='Conv2d_0a_1x1')
								branch_2 = slim.conv3d(branch_2, 64, [3, 3, 3], scope='Conv2d_0b_3x3')
							with tf.variable_scope('Branch_3'):
								branch_3 = slim.max_pool3d(net, [3, 3, 3], scope='MaxPool_0a_3x3')
								branch_3 = slim.conv3d(branch_3, 64, [1, 1, 1], scope='Conv2d_0b_1x1')
							net = tf.concat(axis=4, values=[branch_0, branch_1, branch_2, branch_3])

						with tf.variable_scope('Mixed_4f'):
							with tf.variable_scope('Branch_0'):
								branch_0 = slim.conv3d(net, 256, [1, 1, 1], scope='Conv2d_0a_1x1')
							with tf.variable_scope('Branch_1'):
								branch_1 = slim.conv3d(net, 160, [1, 1, 1], scope='Conv2d_0a_1x1')
								branch_1 = slim.conv3d(branch_1, 320, [3, 3, 3], scope='Conv2d_0b_3x3')
							with tf.variable_scope('Branch_2'):
								branch_2 = slim.conv3d(net, 32, [1, 1, 1], scope='Conv2d_0a_1x1')
								branch_2 = slim.conv3d(branch_2, 128, [3, 3, 3], scope='Conv2d_0b_3x3')
							with tf.variable_scope('Branch_3'):
								branch_3 = slim.max_pool3d(net, [3, 3, 3], scope='MaxPool_0a_3x3')
								branch_3 = slim.conv3d(branch_3, 128, [1, 1, 1], scope='Conv2d_0b_1x1')
							net = tf.concat(axis=4, values=[branch_0, branch_1, branch_2, branch_3])

						net = slim.max_pool3d(net, [2, 2, 2], stride=2, scope='MaxPool_5a_2x2x2')

						with tf.variable_scope('Mixed_5b'):
							with tf.variable_scope('Branch_0'):
								branch_0 = slim.conv3d(net, 256, [1, 1, 1], scope='Conv2d_0a_1x1')
							with tf.variable_scope('Branch_1'):
								branch_1 = slim.conv3d(net, 160, [1, 1, 1], scope='Conv2d_0a_1x1')
								branch_1 = slim.conv3d(branch_1, 320, [3, 3, 3], scope='Conv2d_0b_3x3')
							with tf.variable_scope('Branch_2'):
								branch_2 = slim.conv3d(net, 32, [1, 1, 1], scope='Conv2d_0a_1x1')
								branch_2 = slim.conv3d(branch_2, 128, [3, 3, 3], scope='Conv2d_0a_3x3')
							with tf.variable_scope('Branch_3'):
								branch_3 = slim.max_pool3d(net, [3, 3, 3], scope='MaxPool_0a_3x3')
								branch_3 = slim.conv3d(branch_3, 128, [1, 1, 1], scope='Conv2d_0b_1x1')
							net = tf.concat(axis=4, values=[branch_0, branch_1, branch_2, branch_3])

						with tf.variable_scope('Mixed_5c'):
							with tf.variable_scope('Branch_0'):
								branch_0 = slim.conv3d(net, 384, [1, 1, 1], scope='Conv2d_0a_1x1')
							with tf.variable_scope('Branch_1'):
								branch_1 = slim.conv3d(net, 192, [1, 1, 1], scope='Conv2d_0a_1x1')
								branch_1 = slim.conv3d(branch_1, 384, [3, 3, 3], scope='Conv2d_0b_3x3')
							with tf.variable_scope('Branch_2'):
								branch_2 = slim.conv3d(net, 48, [1, 1, 1], scope='Conv2d_0a_1x1')
								branch_2 = slim.conv3d(branch_2, 128, [3, 3, 3], scope='Conv2d_0b_3x3')
							with tf.variable_scope('Branch_3'):
								branch_3 = slim.max_pool3d(net, [3, 3, 3], scope='MaxPool_0a_3x3')
								branch_3 = slim.conv3d(branch_3, 128, [1, 1, 1], scope='Conv2d_0b_1x1')
							net = tf.concat(axis=4, values=[branch_0, branch_1, branch_2, branch_3])

						with tf.variable_scope('Logits'):
							net = slim.avg_pool3d(net, [2, 7, 7], stride=1, scope='AvgPool_0a_7x7')
							net = slim.dropout(net, keep_prob, scope='Dropout_0b')
							logits = slim.conv3d(net, num_classes, [1, 1, 1], activation_fn=None,
								normalizer_fn=None, scope='Conv2d_0c_1x1')

							logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')

							averaged_logits = tf.reduce_mean(logits, axis=1)

							return averaged_logits