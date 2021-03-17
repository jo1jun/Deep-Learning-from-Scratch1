# coding: utf-8
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('..')                 
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from deep_convnet import DeepConvNet
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = DeepConvNet()  

# 학습 속도가 너무 느리므로 이미 학습된 params를 불러오자. (아래부터 주석처리 후 맨 아래 주석해제)

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# 매개변수 보관
network.save_params("deep_convnet_params.pkl")
print("Saved Network Parameters!")


# # 이미 학습된 params 으로 정확도를 계산

# network.load_params('deep_convnet_params.pkl')
# print(network.accuracy(x_test, t_test))
# #accuracy 가 99.35% 나온다.