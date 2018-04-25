import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

def _numerical_gradient_no_batch(f, x):
    '''이 함수의 경우는 기울기를 구하려고 할 때, 고려를 해야 할 것이
    x가 이전의 differential.py 의 numerical_diff 에서 와는 다르게
    array로 들어온다는 것이. ex) [3.0, 4.9] 따라서, 두 개의 점이 있으니
    2D 평면상의 점이 구해지게 되고 이점을 기준으로하는 기울기를 가지는 직선을
    그리기 위해서는 array 상에 존재하는 각 변수에 일정한 변화를 주어 변화가
    발생 하였을 때의 변화량을 구해야 한다.
    (밑에 적은 말은 정리를 하기 위해 적언 것인데 vector의 개념을 좀더 확실히
    알아야 겠다.)
     이를 위해 x array 상의 하나의 값을 1e-4(0.0001) 만큼 전방, 후방 차분으로
     변화를 주어 x의 변수 갯수만큼의 횟수로 수행을 하게 되면 최종적으로
     grad라는 array의 각 index에 존재하는 값은 x[0] 값에 대해 numerical_diff 을
     구한 값, x[1] ...  x[n] 이 들어가게 된다. 즉 gad에는 x에 존재하는 element들의 갯
     수 만큼 각 지점에서 구한 전방, 후방 차분을 하여 구한 기울기가 나오게 되고, 이 기울기
     를 가지고 직선을 그리게 되면 기울기 평명에서의 직선이 나온다.
    '''
    print("ndim : ", x.ndim)
    h = 1e-4
    #x와 같은 모양의 0으로 채워진 array 생성
    grad = np.zeros_like(x)
    print("inital x : ", x)
    print("Initial grad : ", grad)
    print("size of x  :", x.size)
    print("size of shape :", x.shape)

    for idx in range(x.size):
        tmp_val = x[idx] #ex. 3.0
        x[idx] = float(tmp_val) + h #ex. 3.0001
        fxh1 = f(x) #전방차분
        print("fxh1 : ", fxh1, "x : ", x) #ex,[9.0006001, 0.0], [3,0001, 0.0]

        x[idx] = tmp_val - h
        fxh2 = f(x) #후방차분
        print("fxh2 : ", fxh2, "x : ", x) #ex,[8.99940001,0], [3,0001, 0]
        grad[idx] = (fxh1 - fxh2) / (2*h) #l[6.0045., 0]
        x[idx] = tmp_val
    return grad


#numerical_gradient for array.
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
    # grad = numerical_gradient(function_2, np.array([3.0, 4.9]))
    # print(grad)
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)

    X = X.flatten()
    Y = Y.flatten()

    grad = numerical_gradient(function_2, np.array([X, Y]))

    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1], angles="xy", color="#666666")  # ,headwidth=10,scale=40,color="#444444")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()


