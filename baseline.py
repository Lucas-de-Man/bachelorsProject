import numpy as np
import os


with open('music/music.npy', 'rb') as f:
    piano = np.load(f)
    violin = np.load(f)
    barsize = np.load(f)


class Model():
    def __init__(self, windowsize=256, alpha=0.004, lr=0.0001, dotWeight=0.5, b1=0.9, b2=0.99):
        self.windowsize = windowsize
        self.weights = np.random.normal(0, 10 / windowsize, 2*windowsize + 1)
        self.alpha = alpha

        self.dotWeight = dotWeight
        self.varWeight = 1 - self.dotWeight

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

        self.MSE = 0
        self.total_grad = np.zeros(self.weights.shape)

    def resetGrads(self):
        self.total_grad = np.zeros(self.weights.shape)

    def updateGrads(self, x, start, y, target):
        for i in range(len(x)):
            d = (i + start) % len(x)
            self.total_grad[i] += x[d] * (y - target)
        self.total_grad[2 * self.windowsize] += y - target

    def adam(self):
        self.b1t *= self.b1
        self.b2t *= self.b2

        self.m = self.b1 * self.m + (1 - self.b1) * self.total_grad
        self.v = self.b2 * self.v + (1 - self.b2) * self.total_grad * self.total_grad

        a = self.lr * np.sqrt(1 - self.b2t) / (1 - self.b1t)
        return a * self.m / (np.sqrt(self.v) + 1.e-6)

    def train(self, startP, startV, length):
        losses = np.empty(length - self.windowsize)
        self.resetGrads()
        #setup input
        input = np.empty(2 * self.windowsize)
        start = 0
        for i in range(self.windowsize):
            input[i] = piano[startP + i] + violin[startV + i]
            input[i+self.windowsize] = input[i] * input[i]
        #main loop
        for t in range(length - self.windowsize):
            y = self.weights[2 * self.windowsize]  # add bias first
            for i in range(self.windowsize):
                j = (start + i) % self.windowsize
                y += self.weights[i] * input[j]  # linear terms
                y += self.weights[i + self.windowsize] * input[j + self.windowsize]  # quadratic terms
            #updating running means and gradients
            self.MSE = (y - input[(start + self.windowsize // 2) % self.windowsize]) ** 2
            #save Loss
            losses[t] = self.MSE
            #updating gradients
            self.updateGrads(input, start, y, piano[(startP + t + self.windowsize // 2) % self.windowsize])
            #stepping one step forward
            input[start] = piano[startP + t + self.windowsize] + violin[startV + t + self.windowsize]
            input[start + self.windowsize] = input[start] * input[start]
            start = (start + 1) % self.windowsize
        #self.weights -= self.adam() / (length - self.windowsize)
        self.weights -= self.lr * self.total_grad / (length - self.windowsize)
        return losses

    #for running a trained network
    def forward(self, startP, startV, length):
        out = np.empty((2, length - self.windowsize))
        input = np.empty(2*self.windowsize)
        start = 0
        for i in range(self.windowsize):
            input[i] = piano[startP + i] + violin[startV + i]
            input[i+self.windowsize] = input[i] * input[i]
        for j in range(len(out[0])):
            out[0][j] = self.weights[2*self.windowsize] #add bias first
            for i in range(self.windowsize):
                out[0][j] += self.weights[i] * input[(start + i) % self.windowsize] #linear terms
                out[0][j] += self.weights[i + self.windowsize] * input[(start + i) % self.windowsize + self.windowsize] #quadratic terms
            out[1][j] = input[(start + self.windowsize // 2) % self.windowsize] - out[0][j]
            input[start] = piano[startP + j + self.windowsize] + violin[startV + j + self.windowsize]
            input[start + self.windowsize] = input[start] * input[start]
            start = (start + 1) % self.windowsize
        return out

def arrToDict(arr):
    out = {}
    for e in arr:
        if e[2] == 'int':
            out[e[0]] = int(e[1])
        elif e[2] == 'float':
            out[e[0]] = float(e[1])
    return out

model = Model(128, lr=0.001, dotWeight=0.5)
length = model.windowsize + 1028

rep = 5000
total_losses = np.empty((rep, length - model.windowsize))
for i in range(rep):
    print('step', i + 1, 'of', rep)
    p = 0#np.random.randint(0, (len(piano) - length) // barsize) * barsize
    v = 0#np.random.randint(0, (len(violin) - length) // barsize) * barsize
    losses = model.train(p, v, length)
    print('   mean loss', sum(losses) / len(losses))
    total_losses[i] = losses

#saving

folder = 'models'
if not os.path.exists(folder):
    os.makedirs(folder)

modelNr = 0
if os.path.exists(folder + '/modelNr.txt'):
    with open(folder + '/modelNr.txt', 'r') as f:
        modelNr = int(f.read())

params = np.array([('windowsize', model.windowsize, 'int'), ('alpha', model.alpha, 'float'),
                   ('lr', model.lr, 'float'), ('b1', model.b1, 'float'), ('b2', model.b2, 'float'),
                   ('rep', rep, 'int'), ('length', length, 'int'), ('dotWeight', model.dotWeight, 'float'),
                   ('b1t', model.b1t, 'float'), ('b2t', model.b2t, 'float')])

with open(folder + '/baseline' + str(modelNr) + '.npy', 'wb') as f:
    np.save(f, model.weights)
    np.save(f, total_losses)
    np.save(f, params)
    np.save(f, model.m)
    np.save(f, model.v)

with open(folder + '/modelNr.txt', 'w') as f:
    f.write(str(modelNr + 1))