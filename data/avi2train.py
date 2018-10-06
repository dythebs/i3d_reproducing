import numpy as np 
import cv2
import os
from threading import Thread
import random

video_path = 'UCF-101'
rgb_np_train_path = 'UCF-101_rgb_np_train'
flow_np_train_path = 'UCF-101_flow_np_train'
list_path = 'ucfTrainTestlist'
thread_num = 10

def task(train_videos):
	for video in train_videos:
		rgb_array = []
		cap = cv2.VideoCapture(os.path.join(video_path, video))
		base_name = video.split('.')[0]
		#rgb
		while True:
			ret, frame = cap.read()
			if frame is None:
				break
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			frame = cv2.resize(frame, (224, 224))
			rgb_array.append(frame)
		cap.release()

		rgb_array = np.array(rgb_array)
		frames = rgb_array.shape[0]
		if frames < 65:#多出一帧计算光流使用
			rgb_array = np.repeat(rgb_array, (65//frames+1), axis=0)
		frames = rgb_array.shape[0]
		start = random.randint(0,frames-64)
		np.save(os.path.join(rgb_np_train_path, base_name+'_rgb'), rgb_array[start:start+64])#只保存64帧
		#flow
		flow_array = []
		prvs = cv2.cvtColor(rgb_array[0], cv2.COLOR_RGB2GRAY)
		p_flow = cv2.DualTVL1OpticalFlow_create()
		for i in range(1, 65): #1-64
			next = cv2.cvtColor(rgb_array[i], cv2.COLOR_RGB2GRAY)
			flow = p_flow.calc(prvs, next, None)
			flow[flow > 20] = 20
			flow[flow < -20] = -20
			flow_array.append(flow)
			prvs = next

		flow_array = np.array(flow_array)
		np.save(os.path.join(flow_np_train_path, base_name+'_flow'), flow_array)

if not os.path.exists(rgb_np_train_path):
	os.mkdir(rgb_np_train_path)
if not os.path.exists(flow_np_train_path):
	os.mkdir(flow_np_train_path)

class_list = os.listdir(video_path)
for class_ in class_list:
	if not os.path.exists(os.path.join(rgb_np_train_path, class_)):
		os.mkdir(os.path.join(rgb_np_train_path, class_))
	if not os.path.exists(os.path.join(flow_np_train_path, class_)):
		os.mkdir(os.path.join(flow_np_train_path, class_))

train_videos = []
for file in os.listdir(list_path):
	if file.startswith('trainlist01'):
		with open(os.path.join(list_path, file), 'r') as fp:
			lines = fp.readlines()
			for line in lines:
				train_videos.append(line.split()[0])

threads = []

for i in range(thread_num):
	total = len(train_videos)
	deal_num = total // thread_num
	if i != thread_num -1:
		start = i*deal_num
		end = (i+1)*deal_num
	else:
		start = i*deal_num
		end = total
	t = Thread(target=task, args=(train_videos[start:end],))
	threads.append(t)
	t.start()

for t in threads:
	t.join()