import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


#numerical_gradient for one variable
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y

def show_diff_in_graph():
    def function_1(x):
        return 0.01*x**2 + 0.1*x

    x = np.arange(0.0, 20.0, 0.1)
    y = function_1(x)
    plt.xlabel("x")
    plt.ylabel("f(x)")

    tf = tangent_line(function_1, 5)
    y2 = tf(x)

    tf1 = tangent_line(function_1, 10)
    y3 = tf1(x)

    plt.plot(x, y, x, y2, x, y3)
    plt.show()


if __name__ == "__main__":
    show_diff_in_graph()

