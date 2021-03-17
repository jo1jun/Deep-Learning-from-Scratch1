# coding: utf-8
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('..')
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:  # pickle file 에 저장된 이미 학습된 가중치 매개변수를 load.
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)             # predict 할 때는 없어도 되는 line 이다. 따라서 a3 를 return 해도 결과는 같다.

    return y


x, t = get_data()               # x.shape = (10000, 784) / t.shape = (10000,)
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):         # 10000개의 data 를 하나씩 빼온다.
    y = predict(network, x[i])
    p= np.argmax(y)             # 확률이 가장 높은 원소의 인덱스를 얻는다.
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
