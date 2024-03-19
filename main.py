import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from scipy.io.wavfile import read, write
import sys

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
        self.linear = nn.Linear((2*windowsize + 1), 2)
        self.biasTensor = torch.tensor([1], dtype=torch.float32)

    def forward(self, x):
        #too slow
        x = torch.cat((x, x**2, self.biasTensor))
        return self.linear(x)

def split_sound(input):
    input += np.random.normal(0, 1000, len(input)).astype(np.int16)
    input = input.astype(np.float32) / 4000
    input = Variable(torch.tensor(np.concatenate((np.zeros(model.windowsize), input)), dtype=torch.float32))
    output = (model.forward(input[:model.windowsize]).detach().numpy() * 4000).astype(np.int16)
    output = np.array([[output[0]], [output[1]]])
    for i in range(1, len(input) - model.windowsize - 1):
        y = (model.forward(input[i:i+model.windowsize]).detach().numpy() * 4000).astype(np.int16)
        y = np.transpose([y])
        output = np.concatenate((output, y), axis=1)
    return output[0], output[1]



model = Linear_regression().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

alpha = 0.1
BATCHSIZE = 1
epochs = 30
input = read("midi/wav_comb/comb100-100_7-0.wav")[1]
#adding random noise
write("without_noise.wav", rate=1764, data=input)
input += np.random.normal(0, 1, len(input)).astype(np.int16) #sd of 1000 is good, but for testing keeping it on 1
write("with_noise.wav", rate=1764, data=input)
input = input.astype(np.float32) / 4000
input = Variable(torch.tensor(np.concatenate((np.zeros(model.windowsize), input)), dtype=torch.float32), requires_grad=True)

#not an epoch, just an gradient decent update
for epoch in range(epochs):
    optimizer.zero_grad()
    loss = Variable(torch.tensor(0, dtype=torch.float32), requires_grad=True)
    running_mean = Variable(torch.tensor([0, 0], dtype=torch.float32), requires_grad=True)
    running_variance = Variable(torch.tensor([0, 0], dtype=torch.float32), requires_grad=True)
    running_dot = Variable(torch.tensor(0, dtype=torch.float32), requires_grad=True)
    running_mag = Variable(torch.tensor([1, 1], dtype=torch.float32), requires_grad=True)

    #debug variables
    MSE = torch.tensor(0, dtype=torch.float32)
    VAR = torch.tensor(0, dtype=torch.float32)
    DOT = torch.tensor(0, dtype=torch.float32)

    for batch in range(BATCHSIZE):
        for i in range(len(input) - model.windowsize - 1):
            x = input[i:i+model.windowsize]
            y = model.forward(x)
            running_mean = running_mean * (1 - alpha) + alpha * y
            running_variance = running_variance * (1 - alpha) + alpha * ((y - running_mean) ** 2)
            running_mag = running_mag * (1 - alpha) + alpha * (y ** 2)
            running_dot = running_dot * (1 - alpha) + alpha * y[0] * y[1]
            MSE += (input[i+model.windowsize+1] - y[0] - y[1]) ** 2
            VAR += running_variance[0] * running_variance[1]
            DOT += running_dot ** 2 / (running_mag[0] * running_mag[1])
            loss = loss + (input[i+model.windowsize+1] - y[0] - y[1]) ** 2 - running_variance[0] * running_variance[1] / 10000 + running_dot ** 2 / (running_mag[0] * running_mag[1])
    loss = loss / BATCHSIZE / len(input) / model.windowsize
    print('epoch ', 1+epoch, '/', epochs, ', final loss:', loss.item())
    print(MSE.item(), VAR.item(), DOT.item())
    if epoch != epochs - 1:
        print("updating...")
        loss.backward()
        optimizer.step()
    print("done.")

y0, y1 = split_sound(read("midi/wav_comb/comb100-100_7-0.wav")[1])
write("out0.wav", rate=1764, data=y0)
write("out1.wav", rate=1764, data=y1)
