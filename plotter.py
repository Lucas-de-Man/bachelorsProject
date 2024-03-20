import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.io.wavfile import read

print('plotting...')

MODEL_FOLDER = 'models'
with open(MODEL_FOLDER + "/data.npy", 'rb') as f:
    data = np.load(f)
PIANO = read("midi/wavs/piano.wav")[1][:, 0].astype(np.float32) / 65536 * data[0]
VIOLIN = read("midi/wavs/violin.wav")[1][:, 0].astype(np.float32) / 65536 * data[0]


class Linear_regression(torch.nn.Module):
    def __init__(self, windowsize=256):
        super().__init__()
        self.windowsize = windowsize
        self.linear = torch.nn.Linear((2*windowsize + 1), 2, bias=False)

    def forward(self, x):
        #making the full input at O(n)
        input = torch.empty(2*self.windowsize + 1, dtype=torch.float32)
        input[0] = 1
        for i in range(self.windowsize):
            input[i+1] = x[i]
            input[i + self.windowsize + 1] = x[i] * x[i]
        return self.linear(input)

def plotLosses(subplot):
    print('plotting losses...')
    with open(MODEL_FOLDER + '/losses.npy', 'rb') as f:
        losses = np.load(f)

    subplot.plot(losses[:, 0])
    subplot.plot(losses[:, 1])
    subplot.plot(losses[:, 2])
    subplot.plot(losses[:, 3])

    subplot.legend(["loss", "MSE", "VAR", "DOT"])
    print('done')

def runModel(model : Linear_regression, input):
    out0 = np.empty(len(input) - model.windowsize)
    out1 = np.empty(len(input) - model.windowsize)
    end = len(input) - model.windowsize
    for i in range(end):
        y = model.forward(input[i:i + model.windowsize])
        out0[i] = y[0]
        out1[i] = y[1]
        if i % (end // 5) == 0:
            print(int(i / (end // 5)), '/ 5')
    return out0, out1

def plotChannels(subplot, width=2):
    if width <= 1:
        width = 2
    print("plotting channels...")
    model = Linear_regression()
    model.load_state_dict(torch.load(MODEL_FOLDER + '/model.pt'))
    input = PIANO + VIOLIN
    input = input[:width * model.windowsize]
    c0, c1 = runModel(model, input)
    input = input[model.windowsize // 2: len(input) - model.windowsize // 2]
    subplot[0].plot(input)
    subplot[1].plot(c0)
    subplot[2].plot(c1)
    print('done')

def plotValidatons(subplot, width=5):
    if width <= 1:
        width = 2
    print('plotting validation loss...')
    with open(MODEL_FOLDER + "/times.npy", 'rb') as f:
        times = np.load(f)
    test_in = PIANO + VIOLIN
    models = len(times)
    dots = np.empty(models)
    mags = np.empty(models)
    for i in range(models):
        print('model', i, '/', models)
        model = Linear_regression()
        model.load_state_dict(torch.load(MODEL_FOLDER + '/intermediate/model' + str(i) + '.pt'))
        test_in = test_in[:width * model.windowsize]
        c0, c1 = runModel(model, test_in)
        mag0, mag1 = np.sqrt(sum(c0 ** 2)), np.sqrt(sum(c1 ** 2))
        comp_piano = PIANO[model.windowsize // 2:len(test_in) - model.windowsize // 2]
        piano_mag = np.sqrt(sum(comp_piano ** 2))
        comp_violin = VIOLIN[model.windowsize // 2:len(test_in) - model.windowsize // 2]
        violin_mag = np.sqrt(sum(comp_violin ** 2))

        dot_p0 = sum(c0 * comp_piano) / piano_mag / mag0 + np.dot(c1, comp_violin) / violin_mag / mag1
        dot_p1 = sum(c0 * comp_violin) / violin_mag / mag0 + np.dot(c1, comp_piano) / piano_mag / mag1
        if dot_p0 <= dot_p1:
            dots[i] = dot_p0
            mags[i] = max(mag0 / piano_mag, piano_mag / mag0) + max(mag1 / violin_mag, violin_mag / mag1)
        else:
            dots[i] = dot_p1
            mags[i] = max(mag1 / piano_mag, piano_mag / mag1) + max(mag0 / violin_mag, violin_mag / mag0)
    ax2 = subplot.twinx()
    subplot.set_ylabel("dots", color='blue')
    subplot.plot(times, dots, 'bo')
    ax2.set_ylabel("mags", color='red')
    ax2.plot(times, mags, 'r+')
    print('done')



fig, ax = plt.subplots(2, 3)
plotLosses(ax[0][0])
plotChannels(ax[1])
plotValidatons(ax[0][1], 2)
plt.show()

model = Linear_regression()
model.load_state_dict(torch.load(MODEL_FOLDER + '/model.pt'))

