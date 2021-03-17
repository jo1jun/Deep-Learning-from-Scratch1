# coding: utf-8
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append('..')
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

# 각 가중치의 절대 오차의 평균을 구한다.
for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))

    # 수치적 미분과 해석적 미분의 값의 차이가 거의 없다. 따라서 해석적 미분을 잘 구현한 것!
    # 이렇게 수치적 미분은 매우 느리지만 구현이 쉽고 빨라서 debugging 용도(해석적 미분을 제대로 구현했는 지)로 사용된다.