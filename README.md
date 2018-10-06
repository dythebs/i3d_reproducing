# I3D models

## Overview
This repository contains reproducing code reported in the paper "[Quo Vadis,
Action Recognition? A New Model and the Kinetics
Dataset](https://arxiv.org/abs/1705.07750)" by Joao Carreira and Andrew
Zisserman.

## Running the code

### Setup

Firstly, clone this repository using

`$ git clone https://github.com/deepmind/kinetics-i3d`

Then, download Dataset and List using

```
$ curl -O http://crcv.ucf.edu/data/UCF101/UCF101.rar
$ curl -O http://crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip
````

### Data Preprocessing

Mov uncompressed files to `/data`

`$ python avi2train.py`

### Train 

`$ python inception_v1_3d_train.py`

## Repo Structure
```
.
├── ckpt
│   ├── inception_v1_2016_08_28.tar.gz
│   └── inception_v1.ckpt
├── data
│   ├── UCF-101
│   ├── UCF-101_flow_np_train
│   ├── UCF-101_rgb_np_train
│   ├── avi2train.py
│   └── inception_v1.ckpt
├── ckpt_util.py
├── inception_v1_3d.py
├── inception_v1_3d_train.py
└── input_data.py
```
