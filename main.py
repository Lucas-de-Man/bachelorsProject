import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from scipy.io.wavfile import read
import sys
import os

#for debugging
np.random.seed(50937)

#65536 = 2 ^16, so the maximal possible value, so clamping it between 0 and 1
#now between 0 and ampl
ampl = 20
piano = read("midi/wavs/piano.wav")[1][:, 0].astype(np.float32) / 65536 * ampl
violin = read("midi/wavs/violin.wav")[1][:, 0].astype(np.float32) / 65536 * ampl

class Model():
    def __init__(self, windowsize=256, alpha=0.001, lr=0.001):
        self.windowsize = windowsize
        self.weights = np.random.normal(0, 0.1, 2*windowsize + 1)
        self.alpha = alpha
        self.lr = lr

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

    def updateGrads(self, x, y):
        for d in range(len(x)):
            #mean grads
            self.mean_grad[0][d] = (1 - self.alpha) * self.mean_grad[0][d] + x[d]
            self.mean_grad[1][d] = (1 - self.alpha) * self.mean_grad[1][d] - x[d]
            #mag grads
            self.mag_grad[0][d] = (1 - self.alpha) * self.mag_grad[0][d] + x[d] * y[0]
            self.mag_grad[1][d] = (1 - self.alpha) * self.mag_grad[0][d] - x[d] * y[1]
            #var grads
            self.var_grad[0][d] = (1 - self.alpha) * self.var_grad[0][d] + (y[0] - self.mean[0]) * (x[d] - self.alpha * self.mean_grad[0][d]) * 2
            self.var_grad[1][d] = (1 - self.alpha) * self.var_grad[1][d] - (y[1] - self.mean[1]) * (x[d] - self.alpha * self.mean_grad[1][d]) * 2
            #dot grad
            self.dot_grad[d] = (1 - self.alpha) * self.dot_grad[d] + (y[1] - y[0]) * x[d]
        #biasess
        # mean grads
        d = self.windowsize * 2
        self.mean_grad[0][d] = (1 - self.alpha) * self.mean_grad[0][d] + 1
        self.mean_grad[1][d] = (1 - self.alpha) * self.mean_grad[1][d] - 1
        # mag grads
        self.mag_grad[0][d] = (1 - self.alpha) * self.mag_grad[0][d] + y[0]
        self.mag_grad[1][d] = (1 - self.alpha) * self.mag_grad[0][d] - y[1]
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

    def run(self, startP, startV, length):
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
            #updating gradients
            self.updateGrads(input, y)
            #stepping one step forward
            input[start] = piano[startP + t + self.windowsize] + violin[startV + t + self.windowsize]
            input[start + self.windowsize] = input[start] * input[start]
            start = (start + 1) % self.windowsize
        self.weights -= self.lr * self.total_grad / (length - self.windowsize)

    #for running a trained network
    def forward(self, timeseries):
        out = np.empty((len(timeseries) - self.windowsize, 2))
        input = np.empty(2*self.windowsize)
        start = 0
        for i in range(self.windowsize):
            input[i] = timeseries[i]
            input[i+self.windowsize] = timeseries[i] * timeseries[i]
        for j in range(len(out)):
            out[j][0] = self.weights[2*self.windowsize] #add bias first
            for i in range(self.windowsize):
                out[j][0] += self.weights[i] * input[(start + i) % self.windowsize] #linear terms
                out[j][0] += self.weights[i + self.windowsize] * input[(start + i) % self.windowsize + self.windowsize] #quadratic terms
            out[j][1] = input[(start + self.windowsize // 2) % self.windowsize] - out[j][0]
            input[start] = timeseries[j + self.windowsize]
            input[start + self.windowsize] = timeseries[j + self.windowsize] * timeseries[j + self.windowsize]
            start = (start + 1) % self.windowsize
        return out

#print(model.forward(timeseries))

sys.exit()

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class Linear_regression(nn.Module):
    def __init__(self, windowsize=256):
        super().__init__()
        self.windowsize = windowsize
        self.change = torch.zeros(2*windowsize + 1)
        self.linear = nn.Linear(2*windowsize + 1, 1, bias=False)

    def forward(self, x):
        out = torch.zeros(2)
        #computing linear with the squares
        for i in range(self.windowsize):
            out[0] += self.linear.weight[0][i] * x[i] + self.linear.weight[0][i + self.windowsize] * x[i] * x[i]
        #adding the bias term
        out[0] += self.linear.weight[0][2*self.windowsize]
        #creating chanel 2
        #every weight in chanel 2 should be negative chanel 1, exept at k = windowsize // 2, where they should sum to 1
        #this means w0k + w1k = 1 => w1k = 1 - w0k. After summing all values this is the same as adding xk.
        out[1] = x[self.windowsize // 2] - out[0]
        return out

def updateInput(piano, violin, inputSize):
    input = torch.empty(inputSize, dtype=torch.float32)
    p = np.random.randint(0, len(piano) - inputSize)
    v = np.random.randint(0, len(violin) - inputSize)
    for i in range(inputSize):
        input[i] = piano[p + i] + violin[v + i] #+ np.random.normal(0, 0.02, model.windowsize)
    return input



piano = torch.tensor(piano, dtype=torch.float32)
violin = torch.tensor(violin, dtype=torch.float32)

model = Linear_regression().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if not os.path.exists("models"):
    os.makedirs("models")
if not os.path.exists("models/intermediate"):
    os.makedirs("models/intermediate")

#used hyper parameter search, this alpha works quite well
alpha = 0.005
runs = 100
BATCHSIZE = 50
#used hyper parameter search, 8 was the best but not by a long shot, anything greater than 6 is fine
#do way longer
inputSize = 8 * model.windowsize
losses = np.zeros((runs, 3))
intermediateModels = 20

if False:
    model_name = 'models-100-50-8'
    print("using", model_name)
    optimizer.load_state_dict(torch.load(model_name + '/adam.pt'))
    model.load_state_dict(torch.load(model_name + '/model.pt'))
    with open(model_name + "/losses.npy", 'rb') as f:
        old = np.load(f)
    with open(model_name + "/times.npy", 'rb') as f:
        times = np.load(f)
    startL = len(old)
    startT = len(times)
    oldEnd = times[-1]
    losses = np.concatenate((old, losses))
    for i in range(startT):
        dict = torch.load(model_name + '/intermediate/model' + str(i + 1) + '.pt')
        torch.save(dict, 'models/intermediate/model' + str(i + 1) + '.pt')
else:
    startT = 0
    startL = 0
    oldEnd = 0
    times = np.array([])

if intermediateModels >= runs:
    intermediateModels = min(runs, 4)

print("started.")
for run in range(runs):
    optimizer.zero_grad()
    loss = Variable(torch.zeros(1, dtype=torch.float32), requires_grad=True)
    #expect a mean of 0 in the limit
    running_mean = Variable(torch.zeros(2, dtype=torch.float32), requires_grad=True)
    #in the limit we expect a similar variance to the input, I might implement an expected value
    running_variance = Variable(torch.zeros(2, dtype=torch.float32), requires_grad=True)
    #in the limit we want a dot product of 0
    running_dot = Variable(torch.zeros(1, dtype=torch.float32), requires_grad=True)
    #in the limit we expect a magnitude equal to the mean amplitude, now around 1 (more towards 1/3 but close enough)
    running_mag = Variable(torch.zeros(2, dtype=torch.float32), requires_grad=True)

    #debug variables
    VAR = torch.zeros(1, dtype=torch.float32)
    DOT = torch.zeros(1, dtype=torch.float32)

    for batch in range(BATCHSIZE):
        input = updateInput(piano, violin, inputSize)
        for i in range(inputSize - model.windowsize):
            x = input[i:i+model.windowsize]
            y = model.forward(x)
            running_mean = running_mean * (1 - alpha) + alpha * y
            running_variance = running_variance * (1 - alpha) + alpha * ((y - running_mean) ** 2)
            running_mag = running_mag * (1 - alpha) + alpha * (y ** 2)
            running_dot = running_dot * (1 - alpha) + alpha * y[0] * y[1]
            #normalizing the variance to not let it blow up to infinity
            var = running_variance[0] * running_variance[1] / torch.sqrt(running_mag[0] * running_mag[1])
            #the main goal of the model is to lower the dot product, so I give it a higher weight
            dot = running_dot ** 2 / (running_mag[0] * running_mag[1]) * 5
            VAR += var
            DOT += dot
            loss = loss - var + dot
    loss = loss / BATCHSIZE / (inputSize - model.windowsize)
    print('run ', 1+run, '/', runs, ', loss:', loss.item())
    losses[run + startL][0] = loss.item()
    losses[run + startL][1] = VAR.item() / BATCHSIZE / (inputSize - model.windowsize)
    losses[run + startL][2] = DOT.item() / BATCHSIZE / (inputSize - model.windowsize)
    print("VAR", losses[run + startL][1], "DOT", losses[run + startL][2])
    print("updating...")
    loss.backward()
    optimizer.step()
    print("done.")
    if (run + 1) % (runs // intermediateModels) == 0:
        torch.save(model.state_dict(), "models/intermediate/model" + str(startT + int((run + 1) / (runs // intermediateModels))) + ".pt")
        times = np.append(times, run + oldEnd)


#saving the model and the losses for later analysis
torch.save(model.state_dict(), "models/model.pt")
torch.save(optimizer.state_dict(), "models/adam.pt")

with open('models/losses.npy', 'wb') as f:
    np.save(f, losses)
with open('models/times.npy', 'wb') as f:
    np.save(f, times)
with open('models/data.npy', 'wb') as f:
    np.save(f, np.array([ampl, model.windowsize]))