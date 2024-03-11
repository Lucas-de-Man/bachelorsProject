import torch
import numpy as np
from torch import nn
from torch.autograd import Variable

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
        self.linear = torch.nn.Linear((2*windowsize + 1), 2)
        self.biasTensor = torch.tensor([1], dtype=torch.float32)

    def forward(self, x):
        x = torch.cat((x, x, self.biasTensor))
        return self.linear(x)

model = Linear_regression().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

BATCHSIZE = 10
epochs = 5000
for epoch in range(epochs):
    optimizer.zero_grad()
    loss = Variable(torch.tensor(0, dtype=torch.float32), requires_grad=True)

    if epoch % (epochs // 10) == 0:
        print("info")



