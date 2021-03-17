# coding: utf-8


class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y                
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y  # x와 y를 바꾼다.
        dy = dout * self.x

        return dx, dy


class AddLayer:
    def __init__(self):
        pass  # 덧셈 계층에서는 순전파시 입력값이 역전파에서 필요없다. 따라서 기억할 필요 x

    def forward(self, x, y):
        out = x + y

        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        # 덧셈의 역전파는 상류에서 넘어온 미분을 그대로 전달한다.
        return dx, dy
