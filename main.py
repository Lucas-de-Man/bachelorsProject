import numpy as np
from scipy.io.wavfile import read
from matplotlib import pyplot as plt
import os

#for debugging
np.random.seed(50937)

#65536 = 2 ^16, so the maximal possible value, so clamping it between 0 and 1
#now between 0 and ampl
ampl = 20
piano = read("midi/wavs/piano.wav")[1][:, 0].astype(np.float32) / 65536 * ampl
violin = read("midi/wavs/violin.wav")[1][:, 0].astype(np.float32) / 65536 * ampl

class Model():
    def __init__(self, windowsize=256, alpha=0.001, lr=0.001, b1=0.9, b2=0.99):
        self.windowsize = windowsize
        self.weights = np.random.normal(0, 0.1, 2*windowsize + 1)
        self.alpha = alpha

        #parameters for adam
        self.lr = lr
        #starting values
        self.b1 = b1
        self.b2 = b2
        #decaying values
        self.b1t = 1
        self.b2t = 1
        #first and second order moment estimations
        self.m = np.zeros(len(self.weights))
        self.v = np.zeros(len(self.weights))

        self.mean = np.zeros(2)
        self.mag = np.zeros(2)
        self.var = np.zeros(2)
        self.dot = 0

        self.mean_grad = np.zeros((2,) + self.weights.shape)
        self.mag_grad = np.zeros((2,) + self.weights.shape)
        self.var_grad = np.zeros((2,) + self.weights.shape)
        self.dot_grad = np.zeros(self.weights.shape)
        self.D_grad = np.zeros(self.weights.shape)
        self.V_grad = np.zeros(self.weights.shape)

    def resetGrads(self):
        self.mean_grad = np.zeros((2,) + self.weights.shape)
        self.mag_grad = np.zeros((2,) + self.weights.shape)
        self.var_grad = np.zeros((2,) + self.weights.shape)
        self.dot_grad = np.zeros(self.weights.shape)
        self.total_grad = np.zeros(self.weights.shape)

    def updateGrads(self, x, start, y):
        for i in range(len(x)):
            d = (i + start) % len(x)
            #mean grads
            self.mean_grad[0][i] = (1 - self.alpha) * self.mean_grad[0][i] + x[d]
            self.mean_grad[1][i] = -self.mean_grad[0][d]
            #mag grads
            self.mag_grad[0][i] = (1 - self.alpha) * self.mag_grad[0][i] + x[d] * y[0]
            self.mag_grad[1][i] = (1 - self.alpha) * self.mag_grad[1][i] - x[d] * y[1]
            #dot grad
            self.dot_grad[i] = (1 - self.alpha) * self.dot_grad[i] + (y[1] - y[0]) * x[d]
            #var grads
            self.var_grad[0][i] = (1 - self.alpha) * self.var_grad[0][i] + (y[0] - self.mean[0]) * (x[d] - self.alpha * self.mean_grad[0][i]) * 2
            self.var_grad[1][i] = (1 - self.alpha) * self.var_grad[1][i] - (y[1] - self.mean[1]) * (x[d] - self.alpha * self.mean_grad[1][i]) * 2
        #biasess
        # mean grads
        d = self.windowsize * 2
        self.mean_grad[0][d] = (1 - self.alpha) * self.mean_grad[0][d] + 1
        self.mean_grad[1][d] = (1 - self.alpha) * self.mean_grad[1][d] - 1
        # mag grads
        self.mag_grad[0][d] = (1 - self.alpha) * self.mag_grad[0][d] + y[0]
        self.mag_grad[1][d] = (1 - self.alpha) * self.mag_grad[1][d] - y[1]
        # var grads
        self.var_grad[0][d] = (1 - self.alpha) * self.var_grad[0][d] + (y[0] - self.mean[0]) * (1 - self.alpha * self.mean_grad[0][d]) * 2
        self.var_grad[1][d] = (1 - self.alpha) * self.var_grad[1][d] - (y[1] - self.mean[1]) * (1 - self.alpha * self.mean_grad[1][d]) * 2
        # dot grad
        self.dot_grad[d] = (1 - self.alpha) * self.dot_grad[d] + y[1] - y[0]
        # total grad
        mags = self.mag[0] * self.mag[1]
        diff = (self.mag[0] * self.mag_grad[1] + self.mag[1] * self.mag_grad[0])
        top = (2 * self.dot_grad * self.dot - self.var[0] * self.var_grad[1] - self.var[1] * self.var_grad[0]) * mags
        top += (self.var[0] * self.var[1] - self.dot * self.dot) * diff
        self.total_grad += top / (mags * mags)

    def adam(self):
        self.b1t *= self.b1
        self.b2t *= self.b2

        self.m = self.b1 * self.m + (1 - self.b1) * self.total_grad
        self.v = self.b2 * self.v + (1 - self.b2) * self.total_grad * self.total_grad

        a = self.lr * np.sqrt(1 - self.b2t) / (1 - self.b1t)
        return a * self.m / (np.sqrt(self.v) + 1.e-6)

    def train(self, startP, startV, length, report_loss=True):
        losses = np.empty(length - self.windowsize)
        #reset running averages
        self.mean = np.zeros(2)
        self.mag = np.zeros(2)
        self.var = np.zeros(2)
        self.dot = 0
        self.resetGrads()
        #setup input
        input = np.empty(2 * self.windowsize)
        start = 0
        for i in range(self.windowsize):
            input[i] = piano[startP + i] + violin[startV + i]
            input[i+self.windowsize] = input[i] * input[i]
        #main loop
        for t in range(length - self.windowsize):
            y = np.empty(2)
            y[0] = self.weights[2 * self.windowsize]  # add bias first
            for i in range(self.windowsize):
                y[0] += self.weights[i] * input[(start + i) % self.windowsize]  # linear terms
                y[0] += self.weights[i + self.windowsize] * input[(start + i) % self.windowsize + self.windowsize]  # quadratic terms
            y[1] = input[(start + self.windowsize // 2) % self.windowsize] - y[0]
            #updating running means and gradients
            self.mean = (1 - self.alpha) * self.mean + self.alpha * y
            self.mag = (1 - self.alpha) * self.mag + self.alpha * y * y
            error = y - self.mean
            self.var = (1 - self.alpha) * self.var + self.alpha * error * error
            self.dot = (1 - self.alpha) * self.dot + self.alpha * y[0] * y[1]
            if report_loss:
                losses[t] = (self.dot * self.dot - self.var[0] * self.var[1]) / (self.mag[0] * self.mag[1])
            #updating gradients
            self.updateGrads(input, start, y)
            #stepping one step forward
            input[start] = piano[startP + t + self.windowsize] + violin[startV + t + self.windowsize]
            input[start + self.windowsize] = input[start] * input[start]
            start = (start + 1) % self.windowsize
        self.weights -= self.adam()
        return losses

    #for running a trained network
    def forward(self, startP, startV, length):
        out = np.empty((length - self.windowsize, 2))
        input = np.empty(2*self.windowsize)
        start = 0
        for i in range(self.windowsize):
            input[i] = piano[startP + i] + violin[startV + i]
            input[i+self.windowsize] = input[i] * input[i]
        for j in range(len(out)):
            out[j][0] = self.weights[2*self.windowsize] #add bias first
            for i in range(self.windowsize):
                out[j][0] += self.weights[i] * input[(start + i) % self.windowsize] #linear terms
                out[j][0] += self.weights[i + self.windowsize] * input[(start + i) % self.windowsize + self.windowsize] #quadratic terms
            out[j][1] = input[(start + self.windowsize // 2) % self.windowsize] - out[j][0]
            input[start] = piano[startP + j + self.windowsize] + violin[startV + j + self.windowsize]
            input[start + self.windowsize] = input[start] * input[start]
            start = (start + 1) % self.windowsize
        return out

    #evaluating the current network
    def losses(self, startP, startV, length):
        losses = np.empty(length - self.windowsize)
        input = np.empty(2*self.windowsize)
        start = 0
        for i in range(self.windowsize):
            input[i] = piano[startP + i] + violin[startV + i]
            input[i+self.windowsize] = input[i] * input[i]
        out = np.empty(2)
        mean = np.zeros(2)
        mag = np.zeros(2)
        var = np.zeros(2)
        dot = 0
        for j in range(len(losses)):
            out[0] = self.weights[2*self.windowsize] #add bias first
            for i in range(self.windowsize):
                out[0] += self.weights[i] * input[(start + i) % self.windowsize] #linear terms
                out[0] += self.weights[i + self.windowsize] * input[(start + i) % self.windowsize + self.windowsize] #quadratic terms
            out[1] = input[(start + self.windowsize // 2) % self.windowsize] - out[0]

            mean = (1 - self.alpha) * mean + self.alpha * out
            var = (1 - self.alpha) * var + self.alpha * (out - mean) * (out - mean)
            mag = (1 - self.alpha) * mag + self.alpha * out * out
            dot = (1 - self.alpha) * dot + self.alpha * out[0] * out[1]
            losses[j] = (dot * dot - var[0] * var[1]) / (mag[0] * mag[1])

            input[start] = piano[startP + j + self.windowsize] + violin[startV + j + self.windowsize]
            input[start + self.windowsize] = input[start] * input[start]
            start = (start + 1) % self.windowsize
        return losses

model = Model(1024, lr=10)
rep = 120
length = model.windowsize + 1000
mean_loss = np.empty(rep)
for i in range(rep):
    print(i)
    p = np.random.randint(0, len(piano) - length)
    v = np.random.randint(0, len(violin) - length)
    losses = model.train(p, v, length)
    mean_loss[i] = sum(losses) / len(losses)
plt.plot(mean_loss)
plt.show()