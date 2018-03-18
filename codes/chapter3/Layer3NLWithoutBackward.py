import numpy as np

class Layer3NLWithoutBackward(object):
    def __init__(self, X):
        self.network = {}
        self.xarray = X

    def sigmoid(self, array):
        return 1/(1+np.exp(-array))

    def identical(selfs, array):
        return array

    def initNeuralNetwork(self):
        self.network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
        self.network['B1'] = np.array([0.1, 0.2, 0.3])
        self.network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
        self.network['B2'] = np.array([0.1, 0.2])
        self.network['W3'] = np.array([[0.1, 0.3],[0.2, 0.4]])
        self.network['B3'] = np.array([0.1, 0.2])

    def forward(self):
        W1, W2, W3 = self.network['W1'], self.network['W2'], self.network['W3']
        B1, B2, B3 = self.network['B1'], self.network['B2'], self.network['B3']

        A1 =np.dot(self.xarray,W1) + B1
        Z1 = self.sigmoid(A1)
        A2 = np.dot(Z1,W2) + B2
        Z2 = self.sigmoid(A2)
        A3 = np.dot(Z2, W3) + B3
        Z3 = self.identical(A3)
        return Z3

if __name__ == "__main__":
   nl3 = Layer3NLWithoutBackward(np.array([1.0, 0.5]))
   nl3.initNeuralNetwork()
   print(nl3.forward())


