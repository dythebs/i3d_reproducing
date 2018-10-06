import zipfile
import os
import requests
import numpy as np 
from sklearn import preprocessing
import tensorflow as tf 

url = 'http://crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip'
data_path = 'data'
list_path = os.path.join(data_path, 'ucfTrainTestlist')
list_file = 'trainlist01'
class_file = 'classInd.txt'

video_path = os.path.join(data_path, 'UCF-101')
rgb_np_train_path = os.path.join(data_path, 'UCF-101_rgb_np_train')
flow_np_train_path = os.path.join(data_path, 'UCF-101_flow_np_train')

def maybedownload(url):
	if not os.path.exists(data_path):
		os.mkdir(data_path)
	filename = url.split('/')[-1]
	if not os.path.exists(os.path.join(data_path, filename)):
		with open(os.path.join(data_path, filename), 'wb') as fp:
			fp.write(requests.get(url).content)

	with zipfile.ZipFile(os.path.join(data_path, filename)) as fp:
		fp.extractall(path=data_path)

def paser(rgb_filename, flow_filename, label):
	return np.load(rgb_filename.decode()), np.load(flow_filename.decode()), label.decode()


def read_data(sess, batch_size=6, epeoch=10):

	#获取列表中数据的地址和标签
	rgb_filenames = []
	flow_filenames = []
	labels = []
	for file in os.listdir(list_path):
		if file.startswith(list_file):
			with open(os.path.join(list_path, file), 'r') as fp:
				lines = fp.readlines()
				for line in lines:
					rgb_filenames.append(os.path.join(rgb_np_train_path ,line.split()[0][:-4]+'_rgb.npy'))
					flow_filenames.append(os.path.join(flow_np_train_path, line.split()[0][:-4]+'_flow.npy'))
					labels.append(line.split()[0].split('/')[0])

	#创建数据集
	rgb_filenames_ph = tf.placeholder(dtype=tf.string, shape=[None])
	flow_filenames_ph = tf.placeholder(dtype=tf.string, shape=[None])
	labels_ph = tf.placeholder(dtype=tf.string, shape=[None])

	dataset = tf.data.Dataset.from_tensor_slices((rgb_filenames, flow_filenames, labels))
	dataset = dataset.map(lambda rgb_filename, flow_filename, label:tuple(tf.py_func(
		paser, [rgb_filename, flow_filename, label], [tf.uint8, tf.float32, tf.string])))
	dataset = dataset.shuffle(500).batch(batch_size).repeat()
	iterator = dataset.make_initializable_iterator()
	one_element = iterator.get_next()
	sess.run(iterator.initializer, feed_dict={rgb_filenames_ph:rgb_filenames, flow_filenames_ph:flow_filenames, labels_ph:labels})

	return one_element


#下载list文件
maybedownload(url)
# OneHot编码
classes = []
with open(os.path.join(list_path, class_file), 'r') as fp:
	for line in fp.readlines():
		classes.append(line.split()[1])

le = preprocessing.LabelEncoder()
le.fit(classes)
oe = preprocessing.OneHotEncoder()
oe.fit(le.transform(classes).reshape(-1, 1))

def one_hot(labels):
	return oe.transform(le.transform(labels).reshape(-1, 1)).toarray()