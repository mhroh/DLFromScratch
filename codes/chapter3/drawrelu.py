import numpy as np
import matplotlib.pylab as plt

def relu(x):
    return np.maximum(0, x)

def draw_graph(x, y):
    plt.plot(x, y)
    plt.ylim(0, 5.0)
    plt.show()


if __name__ == "__main__":
    x = np.arange(-5.0, 5.0, 0.1)
    y = relu(x)
    print(y)

    draw_graph(x, y)

