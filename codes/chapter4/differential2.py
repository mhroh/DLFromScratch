import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

def _numerical_gradient_no_batch(f, x):
    print("ndim : ", x.ndim)
    h = 1e-4
    #x와 같은 모양의 0으로 채워진 array 생성
    grad = np.zeros_like(x)
    print("inital x : ", x)
    print("Initial grad : ", grad)
    print("size of x  :", x.size)
    print("size of shape :", x.shape)

    for idx in range(x.size):
        #각 x값을 임시로 가져옴
        tmp_val = x[idx]
        #임시로 가져온 값에 대해 전방차분 만큼의 값을 더해
        #array의 x index에 대입을 함.
        x[idx] = float(tmp_val) + h
        #전방차분과 아래에서 후방차분을 계산하는데
        #x
        fxh1 = f(x) #전방차분

        x[idx] = tmp_val - h
        fxh2 = f(x) #후방차분
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val



def numerical_gradient(f, x):
    if x.ndim == 1:
        return _numerical_gradient_no_batch(f, x)
    else:
        grad = np.zeros_like(x)

        for idx, x in enumerate(x):
            grad[idx] = _numerical_gradient_no_batch(f, x)

        return grad

def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)

def tangent_line(f, x):
    d = numerical_gradient(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y

if __name__ == '__main__':
    #TODO
    # numpy 에서의 shape, axis, rank 에 대해 정확하게 이해하자
    grad = numerical_gradient(function_2, np.array([3.0, 4.9]))
    print(grad)
    # x0 = np.arange(-2, 2.5, 0.25)
    # x1 = np.arange(-2, 2.5, 0.25)
    # X, Y = np.meshgrid(x0, x1)

    # X = X.flatten()
    # Y = Y.flatten()
    #
    # grad = numerical_gradient(function_2, np.array([X, Y]))
    #
    # plt.figure()
    # plt.quiver(X, Y, -grad[0], -grad[1], angles="xy", color="#666666")  # ,headwidth=10,scale=40,color="#444444")
    # plt.xlim([-2, 2])
    # plt.ylim([-2, 2])
    # plt.xlabel('x0')
    # plt.ylabel('x1')
    # plt.grid()
    # plt.legend()
    # plt.draw()
    # plt.show()
    #

