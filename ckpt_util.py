import requests
import os
import tarfile
from tensorflow.python import pywrap_tensorflow
import tensorflow as tf 
import numpy as np 

url ='http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz'
ckpt_path = 'ckpt'

def maybedownload(url):
	if not os.path.exists(ckpt_path):
		os.mkdir(ckpt_path)
	filename = url.split('/')[-1]
	if not os.path.exists(os.path.join(ckpt_path, filename)):
		with open(os.path.join(ckpt_path, filename), 'wb') as fp:
			fp.write(requests.get(url).content)

	with tarfile.open(os.path.join(ckpt_path, filename)) as fp:
		fp.extractall(path=ckpt_path)

def restore(sess):
	#先检查ckpt文件
	maybedownload(url)
	variables = [v for v in tf.model_variables()]

	#通过2d模型的变量名寻找计算图中对应3d模型的变量
	def get_3d_tensor(name):
		if not name.startswith('global_step') and not name.startswith('InceptionV1/Logits'):
			name_3d = 'InceptionV1_3d' + key[11:]
			for v in variables:
				if v.name.startswith('rgb/'+name_3d):
					rgb_tensor_3d = v
				if v.name.startswith('flow/'+name_3d):
					flow_tensor_3d = v
			return rgb_tensor_3d, flow_tensor_3d
		else:
			return None
	
	checkpoint_path = os.path.join(ckpt_path, "inception_v1.ckpt")
	# 读取ckpt文件
	reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
	var_to_shape_map = reader.get_variable_to_shape_map()

	#遍历文件中所有变量
	for key in var_to_shape_map:
		rgb_tensor_3d, flow_tensor_3d = get_3d_tensor(key)
		if rgb_tensor_3d is None or flow_tensor_3d is None:
			continue
		if 'weight' in key:
			weight = reader.get_tensor(key)
			dims = weight.shape[0]
			weight_3d = np.repeat(weight[np.newaxis, :], dims, axis=0) / dims
			sess.run(tf.assign(rgb_tensor_3d, weight_3d))
			#sess.run(tf.assign(flow_tensor_3d, weight_3d))
		else:
			value = reader.get_tensor(key)
			sess.run(tf.assign(rgb_tensor_3d, value))
			#sess.run(tf.assign(flow_tensor_3d, value))