import os, sys
sys.path.append(os.pardir)
from common.functions import softmax
from common.functions import sigmoid
import numpy as np
from dataset.mnist import load_mnist
from dataset.mnist import pickle

def get_data():
    """normallize image pixel' vaule normalizing. flatten 28by28 to 784 array"""
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    print("x-Shape : ", x.shape)
    print("W1-Shape : ", W1.shape)
    print("b1-Shape : ", b1.shape)
    print("W2-Shape : ", W2.shape)
    print("b2-Shape : ", b2.shape)
    print("W3-Shape : ", W3.shape)
    print("b3-Shape : ", b3.shape)

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y

if __name__=='__main__':
    x, t = get_data()
    network = init_network()

    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        print(y)
        p = np.argmax(y)
        if p == t[i]:
            accuracy_cnt += 1

    print("Accyracy: " + str(float(accuracy_cnt)/ len(x)))



