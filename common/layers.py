# coding: utf-8
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common.functions import *
from common.util import im2col, col2im


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        # dx 는 x 와 size 가 같아야 한다. (이러한 activation function 은 사이즈 변동x 따라서 dout size = dx size)
        # true(1)인 것들만 dout 을 곱해서 흘려보낸다.
        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:  # X @ W + b (행렬 곱 & 더하기) layer
    def __init__(self, W, b):
        self.W = W  # (n1 x n2)
        self.b = b  # (1 x n2)
        
        self.x = None  # (m x n1)
        self.original_x_shape = None
        # 가중치와 편향 매개변수의 미분
        self.dW = None  # (n1 x n2)
        self.db = None  # (1 x n2)

    def forward(self, x):
        # 텐서 대응
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b
        # 2차원인 경우 (m x n1) * (n1 x n2) + (1 x n2) = (m x n2) : out & dout size
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)  # (m x n1) = (m x n2) * (n2 x n1)
        self.dW = np.dot(self.x.T, dout)  # (n1 x n2) = (n1 x m) * (m x n2)
        self.db = np.sum(dout, axis=0)  # (1 x n2) = (m x n2) -> sum -> (1 x n2)
        
        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)
        return dx
    
    # out 의 size 를 알고 dx, dW 의 사이즈는 x, W 사이즈와 같은 것을 이용하여 적절히 행렬곱을 맞추자.

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 손실함수
        self.y = None    # softmax의 출력
        self.t = None    # 정답 레이블(원-핫 인코딩 형태)
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx


class Dropout:
    """
    http://arxiv.org/abs/1207.0580
    """
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True): # train 할 때(train_flg 가 Ture 인 경우)만 dropout 사용. 
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio # forward 때마다 매번 랜덤으로 뉴런 선택
                                                                      # *x.shape 는 튜플로 된 원소들을 unpacking
            return x * self.mask # random 하게 true 로 된 원소들만 forward
        else:
            return x * (1.0 - self.dropout_ratio) # 안 곱해도 된다. 실제 딥러닝 프레임워크들은 비율을 곱하지 x

    def backward(self, dout):
        return dout * self.mask # relu 와 같은 방식으로 동작한다. mask 가 true 인 것만 backward


class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # 합성곱 계층은 4차원, 완전연결 계층은 2차원  

        # 시험할 때 사용할 평균과 분산
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # backward 시에 사용할 중간 데이터
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W # filter 가중치
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # 중간 데이터（backward 시 사용）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 가중치와 편향 매개변수의 기울기
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape # filter 의 size (filter 개수, channel 수, filter 높이, filter 너비)
        N, C, H, W = x.shape # data 의 size (batch 크기, channel 수, data 높이, data 너비)
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad) # input 과 filter 을 각각 unrolling 하고 행렬곱(합성곱)
        col_W = self.W.reshape(FN, -1).T # reshape(FN, -1) 은 FN * 적절한 수 = 기존 원소 수 로 나뉜다. 
                                         # data 와 행렬곱을 할 수 있게 transpose.

        out = np.dot(col, col_W) + self.b # 합성곱
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2) # 나온 output 을 rolling (다음 계층 input) 
        # transpose(0,3,1,2) 0번(N)을 0번, 3번(C)을 1번, 
        # 1번(out_h)을 2번, 2번(out_w)를 3번으로 축 순서 변경(x와 축 동일)
        
        self.x = x
        self.col = col
        self.col_W = col_W
        # 원본, unrolling data, weight 중간데이터로 저장.
        return out

    def backward(self, dout): # col2im 을 사용한다는 것을 제외하면 Affine layer 의 backward 와 똑같다.
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling: # (max)
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad) # input data 를 2차원으로 unrolling
        col = col.reshape(-1, self.pool_h*self.pool_w) # 적절한 수 * pooling filter size = 기존 원소 수
                                                       # 이렇게 shape을 바꿔서 pooling 연산을 쉽게 (max만 써서) 구현할 수 있다.
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2) # 다시 원래 형상으로 rolling

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx
