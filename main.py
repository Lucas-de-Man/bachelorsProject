import numpy as np
import os

#65536 = 2 ^16, so the maximal possible value, so clamping it between 0 and 1
#now between 0 and ampl
ampl = 100
#piano = read("midi/wavs/piano.wav")[1][:, 0].astype(np.float32) / 65536 * ampl
#violin = read("midi/wavs/violin.wav")[1][:, 0].astype(np.float32) / 65536 * ampl
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
            self.mag_grad[0][i] = (1 - self.alpha) * self.mag_grad[0][i] + x[d] * y[0] * 2
            self.mag_grad[1][i] = (1 - self.alpha) * self.mag_grad[1][i] - x[d] * y[1] * 2
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
        self.mag_grad[0][d] = (1 - self.alpha) * self.mag_grad[0][d] + y[0] * 2
        self.mag_grad[1][d] = (1 - self.alpha) * self.mag_grad[1][d] - y[1] * 2
        # var grads
        self.var_grad[0][d] = (1 - self.alpha) * self.var_grad[0][d] + (y[0] - self.mean[0]) * (1 - self.alpha * self.mean_grad[0][d]) * 2
        self.var_grad[1][d] = (1 - self.alpha) * self.var_grad[1][d] - (y[1] - self.mean[1]) * (1 - self.alpha * self.mean_grad[1][d]) * 2
        # dot grad
        self.dot_grad[d] = (1 - self.alpha) * self.dot_grad[d] + y[1] - y[0]
        # total grad
        mags = self.mag[0] * self.mag[1]
        diff = self.mag[0] * self.mag_grad[1] + self.mag[1] * self.mag_grad[0]
        top = (2 * self.dot_grad * self.dot * self.dotWeight - self.varWeight * (self.var[0] * self.var_grad[1] + self.var[1] * self.var_grad[0])) * mags
        top += (self.var[0] * self.var[1] * self.varWeight - self.dot * self.dot * self.dotWeight) * diff
        self.total_grad += top / (mags * mags)

    def adam(self):
        self.b1t *= self.b1
        self.b2t *= self.b2

        self.m = self.b1 * self.m + (1 - self.b1) * self.total_grad
        self.v = self.b2 * self.v + (1 - self.b2) * self.total_grad * self.total_grad

        a = self.lr * np.sqrt(1 - self.b2t) / (1 - self.b1t)
        return a * self.m / (np.sqrt(self.v) + 1.e-6)

    def train(self, startP, startV, length, report_loss=True):
        losses = np.empty((2, length - self.windowsize))
        #reset running averages
        self.mean = np.zeros(2)
        self.mag = np.zeros(2) + 1e-6
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
                j = (start + i) % self.windowsize
                y[0] += self.weights[i] * input[j]  # linear terms
                y[0] += self.weights[i + self.windowsize] * input[j + self.windowsize]  # quadratic terms
            y[1] = input[(start + self.windowsize // 2) % self.windowsize] - y[0]
            #updating running means and gradients
            self.mean = (1 - self.alpha) * self.mean + self.alpha * y
            self.mag = (1 - self.alpha) * self.mag + self.alpha * y * y
            error = y - self.mean
            self.var = (1 - self.alpha) * self.var + self.alpha * error * error
            self.dot = (1 - self.alpha) * self.dot + self.alpha * y[0] * y[1]
            if report_loss:
                losses[0][t] = self.dot * self.dot / (self.mag[0] * self.mag[1])
                losses[1][t] = -self.var[0] * self.var[1] / (self.mag[0] * self.mag[1])
            #updating gradients
            self.updateGrads(input, start, y)
            #stepping one step forward
            input[start] = piano[startP + t + self.windowsize] + violin[startV + t + self.windowsize]
            input[start + self.windowsize] = input[start] * input[start]
            start = (start + 1) % self.windowsize
        self.weights -= self.adam() / (length - self.windowsize)
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
        mag = np.zeros(2) + 1e-6
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

#for debugging
#np.random.seed(50937)

def arrToDict(arr):
    out = {}
    for e in arr:
        if e[2] == 'int':
            out[e[0]] = int(e[1])
        elif e[2] == 'float':
            out[e[0]] = float(e[1])
    return out

if __name__ == "__main__":
    reused = True
    if reused:
        with open('models/unaligned.npy', 'rb') as f:
            weights = np.load(f)
            startLosses = np.load(f)
            params = np.load(f)
            m = np.load(f)
            v = np.load(f)
        params = arrToDict(params)
        model = Model(windowsize=params['windowsize'], alpha=params['alpha'], lr=params['lr'],
                      dotWeight=params['dotWeight'], b1=params['b1'], b2=params['b2'])
        model.b1t = params['b1t']
        model.b2t = params['b2t']
        model.m = m
        model.v = v
        model.weights = weights
        length = params['length']
    else:
        model = Model(128, lr=0.2, dotWeight=0.5)
        length = model.windowsize + 4096

    rep = 1500
    total_losses = np.empty((rep, 2, length - model.windowsize))
    for i in range(rep):
        print('step', i + 1, 'of', rep)
        p = np.random.randint(0, (len(piano) - length) // barsize) * barsize
        v = np.random.randint(0, (len(violin) - length) // barsize) * barsize
        losses = model.train(p, v, length)
        dotloss = sum(losses[0]) / len(losses[0])
        varloss = sum(losses[1]) / len(losses[1])
        print('   mean losses (var, dot, total)', varloss, dotloss, dotloss + varloss)
        total_losses[i] = losses

    if reused:
        total_losses = np.append(startLosses, total_losses, axis=0)
        rep += params['rep']

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
                       ('ampl', ampl, 'int'), ('b1t', model.b1t, 'float'), ('b2t', model.b2t, 'float')])

    with open(folder + '/data' + str(modelNr) + '.npy', 'wb') as f:
        np.save(f, model.weights)
        np.save(f, total_losses)
        np.save(f, params)
        np.save(f, model.m)
        np.save(f, model.v)

    with open(folder + '/modelNr.txt', 'w') as f:
        f.write(str(modelNr + 1))