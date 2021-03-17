# coding: utf-8
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from and_gate import AND
from or_gate import OR
from nand_gate import NAND

# 단일 layer perceptron은 XOR 이 불가능하다.. 비선형 영역을 선형으로 구분할 수 없기 때문이다.

# 그러나 다중 layer perceptron 으로 XOR 가능하다!! 0층에는 NAND 와 OR, 1층에는 AND 로 층을 쌓아 XOR 구현해보자.

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

if __name__ == '__main__':
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = XOR(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))