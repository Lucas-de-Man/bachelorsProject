import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from scipy.io.wavfile import read

np.random.seed(5974)
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
        x = torch.cat((x, x, self.biasTensor))
        return self.linear(x)

model = Linear_regression().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

alpha = 0.1
BATCHSIZE = 1
epochs = 1
input = read("midi/wav_comb/comb10-100_0-0.wav")[1]
#adding random noise
input += np.random.normal(0, 1000, len(input)).astype(np.int16)
input = np.concatenate((np.zeros(model.windowsize), input))
for epoch in range(epochs):
    optimizer.zero_grad()
    loss = Variable(torch.tensor(0, dtype=torch.float32), requires_grad=True)
    running_mean = Variable(torch.tensor([0, 0], dtype=torch.float32), requires_grad=True)
    running_variance = Variable(torch.tensor([0, 0], dtype=torch.float32), requires_grad=True)
    running_dot = Variable(torch.tensor(0, dtype=torch.float32), requires_grad=True)
    for batch in range(BATCHSIZE):
        for i in range(len(input) - model.windowsize - 1):
            x = torch.tensor(input[i:i+model.windowsize], dtype=torch.float32)
            y = model.forward(x)
            running_mean = running_mean * (1 - alpha) + alpha * y
            running_variance = running_variance * (1 - alpha) + alpha * ((y - running_mean) ** 2)
            running_dot = running_dot * (i - alpha) + alpha * y[0] * y[1]
            loss = loss + (input[i+model.windowsize+1] - y[0] - y[1]) ** 2 - running_variance + running_dot


    #if epoch % (epochs // 10) == 0:
    #    print("info")



