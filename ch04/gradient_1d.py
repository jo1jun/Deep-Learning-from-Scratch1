# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def numerical_diff(f, x):
    # h = 10e-50                        # rounding error(반올림 오차)
    h = 1e-4 # 0.0001
    # return (f(x+h) - f(x)) / h        # 실제 미분값하고 오차가 존재.
    return (f(x+h) - f(x-h)) / (2*h)    # 위의 오차를 줄이기 위해 개선.


def function_1(x):
    return 0.01*x**2 + 0.1*x 


def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y
     
x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 5)
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.show()
