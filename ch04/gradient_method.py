# coding: utf-8
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))        
import numpy as np
import matplotlib.pylab as plt
from gradient_2d import numerical_gradient


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append( x.copy() )

        grad = numerical_gradient(f, x) # 수치적 미분은 해석적 미분을 올바르게 구현했는 지 점검용으로 사용한다.
        x -= lr * grad

    return x, np.array(x_history)


def function_2(x):
    return x[0]**2 + x[1]**2


init_x = np.array([-3.0, 4.0])    
# learning rate 가 너무 크다면 좋은 결과를 얻을 수 없다. (발산해버린다.)
lr = 10
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)
print('lr = 10.0')
print(function_2(x))

init_x = np.array([-3.0, 4.0])    
# learning rate 가 너무 작다면 좋은 결과를 얻을 수 없다. (갱신 정도가 매우 작다.)
lr = 1e-10
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)
print('lr = 1e-10')
print(function_2(x))

init_x = np.array([-3.0, 4.0])    
# learning rate 가 너무 크다면 좋은 결과를 얻을 수 없다. (발산해버린다.)
lr = 0.1
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)
print('lr = 0.1')
print(function_2(x))

plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()
