# coding: utf-8
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('..')                 
import numpy as np
import matplotlib.pyplot as plt
from deep_convnet import DeepConvNet
from dataset.mnist import load_mnist

# 수행 시간 비교를 위해 time import
import time

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = DeepConvNet()
network.load_params("deep_convnet_params.pkl")

sampled = 10000 # 고속화를 위한 표본추출
x_test = x_test[:sampled]
t_test = t_test[:sampled]

start = time.time()
print("caluculate accuracy (float64) ... ")
print(network.accuracy(x_test, t_test))
print("time : ", time.time() - start)

# float16(반정밀도)로 형변환
x_test = x_test.astype(np.float16)
for param in network.params.values():
    param[...] = param.astype(np.float16)

start = time.time()
print("caluculate accuracy (float16) ... ")
print(network.accuracy(x_test, t_test))
print("time : ", time.time() - start)

# float16 으로 형 변환시 속도가 더 빨라졌다.
