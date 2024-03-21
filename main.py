import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from scipy.io.wavfile import read
import sys
import os

#for debugging
np.random.seed(50937)

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
        self.linear = nn.Linear((2*windowsize + 1), 2, bias=False)

    def forward(self, x):
        #making the full input at O(n)
        input = torch.empty(2*self.windowsize + 1, dtype=torch.float32)
        input[0] = 1
        for i in range(self.windowsize):
            input[i+1] = x[i]
            input[i + self.windowsize + 1] = x[i] * x[i]
        return self.linear(input)

def updateInput(piano, violin, inputSize):
    input = torch.empty(inputSize, dtype=torch.float32)
    p = np.random.randint(0, len(piano) - inputSize)
    v = np.random.randint(0, len(violin) - inputSize)
    for i in range(inputSize):
        input[i] = piano[p + i] + violin[v + 1] #+ np.random.normal(0, 0.02, model.windowsize)
    return input

#65536 = 2 ^16, so the maximal possible value, so clamping it between 0 and 1
#now between 0 and ampl
ampl = 20
piano = read("midi/wavs/piano.wav")[1][:, 0].astype(np.float32) / 65536 * ampl
violin = read("midi/wavs/violin.wav")[1][:, 0].astype(np.float32) / 65536 * ampl

piano = torch.tensor(piano, dtype=torch.float32)
violin = torch.tensor(violin, dtype=torch.float32)

model = Linear_regression().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if not os.path.exists("models"):
    os.makedirs("models")
if not os.path.exists("models/intermediate"):
    os.makedirs("models/intermediate")

#used hyper parameter search, this alpha works quite well
alpha = 0.01
runs = 100
BATCHSIZE = 50
#used hyper parameter search, 8 was the best but not by a long shot, anything greater than 6 is fine
inputSize = 8 * model.windowsize
losses = np.zeros((runs, 4))
intermediateModels = 20

if False:
    model_name = 'models-500-50-2'
    optimizer.load_state_dict(torch.load(model_name + '/adam.pt'))
    model.load_state_dict(torch.load(model_name + '/model.pt'))
    with open(model_name + "/losses.npy", 'rb') as f:
        old = np.load(f)
    with open(model_name + "/times.npy", 'rb') as f:
        times = np.load(f)
    losses = np.concatenate((old, losses))
else:
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
    MSE = torch.zeros(1, dtype=torch.float32)
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
            #input[i + model.windowsize // 2] to have it estimate the middle sample of the input
            mse = (input[i + model.windowsize // 2] - y[0] - y[1]) ** 2 / (ampl * ampl)
            #maybe use an sigmoid around the variance, to limit the value of brining it to infinity
            var = torch.sigmoid(running_variance[0] * running_variance[1])
            #the main goal of the model is to lower the dot product, so I give it a higher weight
            dot = running_dot ** 2 / (running_mag[0] * running_mag[1]) * 5
            MSE += mse
            VAR += var
            DOT += dot
            loss = loss + mse - var + dot
    loss = loss / BATCHSIZE / (inputSize - model.windowsize)
    print('run ', 1+run, '/', runs, ', loss:', loss.item())
    losses[run][0] = loss.item()
    losses[run][1] = MSE.item() / BATCHSIZE / (inputSize - model.windowsize)
    losses[run][2] = VAR.item() / BATCHSIZE / (inputSize - model.windowsize)
    losses[run][3] = DOT.item() / BATCHSIZE / (inputSize - model.windowsize)
    print("MSE", losses[run][1], "VAR", losses[run][2], "DOT", losses[run][3])
    print("updating...")
    loss.backward()
    optimizer.step()
    print("done.")
    if (run + 1) % (runs // intermediateModels) == 0:
        torch.save(model.state_dict(), "models/intermediate/model" + str(int((run + 1) / (runs // intermediateModels))) + ".pt")
        times = np.append(times, run)


#saving the model and the losses for later analysis
torch.save(model.state_dict(), "models/model.pt")
torch.save(optimizer.state_dict(), "models/adam.pt")

with open('models/losses.npy', 'wb') as f:
    np.save(f, losses)
with open('models/times.npy', 'wb') as f:
    np.save(f, times)
with open('models/data.npy', 'wb') as f:
    np.save(f, np.array([ampl, model.windowsize]))
