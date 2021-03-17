# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    return np.array(x > 0, dtype=np.int) # bool type(True 는 1, False 는 0) numpy array를 int type numpy array로 변환

X = np.arange(-5.0, 5.0, 0.1) # -5 <= ~ < 5 까지 등차 0.1 로 numpy array 생성
Y = step_function(X)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1)  # y축 범위를 -0.1 ~ 1.1 로 지정.
plt.show()
