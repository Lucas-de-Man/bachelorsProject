import numpy as np

class solveLinearReg:
    def __init__(self, windowsize=256):
        self.windowsize = windowsize
        self.solution = np.empty(windowsize + 1)
        self.xxSum = np.zeros((windowsize + 1, windowsize + 1))
        self.xySum = np.zeros((1, windowsize + 1))

    def addStep(self, signal, target):
        target = target[self.windowsize // 2:-self.windowsize + self.windowsize // 2 + 1]
        inputMat = np.empty((len(signal) - self.windowsize + 1, self.windowsize + 1))
        #first line of matrix set
        inputMat[0][self.windowsize] = 1
        for i in range(self.windowsize):
            inputMat[0][i] = signal[i]

        for i in range(1, inputMat.shape[0]):
            #bias
            inputMat[i][self.windowsize] = 1
            #shift values
            for j in range(self.windowsize - 1):
                inputMat[i][j] = inputMat[i - 1][j + 1]
            #set the new values in
            inputMat[i][self.windowsize - 1] = signal[self.windowsize - 1 + i]

        self.xxSum += np.dot(inputMat.transpose(), inputMat)
        self.xySum += np.dot(inputMat.transpose(), target)

    def solve(self):
        return np.dot(self.xySum, np.linalg.inv(self.xxSum))

class Regression:
    def __init__(self, weights, bias):
        self.windowsize = len(weights)
        self.weights = weights
        self.bias = bias

    def forward(self, signal):
        return self.bias + np.convolve(self.weights, signal, 'valid')

    def mse(self, signal, target):
        result = self.forward(signal)
        diff = result - target[self.windowsize // 2:-self.windowsize + self.windowsize // 2 + 1]
        return sum(diff ** 2) / len(diff)