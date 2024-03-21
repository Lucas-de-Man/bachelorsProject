import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.io.wavfile import read, write

print('plotting...')

MODEL_FOLDER = 'models'
with open(MODEL_FOLDER + "/data.npy", 'rb') as f:
    data = np.load(f)
ampl = data[0]
windowsize = 256#data[1]
PIANO = read("midi/wavs/piano.wav")[1][:, 0].astype(np.float32) / 65536 * ampl
VIOLIN = read("midi/wavs/violin.wav")[1][:, 0].astype(np.float32) / 65536 * ampl



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

def runModel(model : Linear_regression, input, prints=5):
    out0 = np.empty(len(input) - windowsize)
    out1 = np.empty(len(input) - windowsize)
    end = len(input) - windowsize
    for i in range(end):
        y = model.forward(input[i:i + windowsize])
        out0[i] = y[0]
        out1[i] = y[1]
        if i % (end // prints) == 0:
            print(int(i / (end // prints)), '/', prints)
    return out0, out1

def plotChannels(subplot, width=2, start=10):
    if start <= 0.9:
        start = 0.9
    if width <= 1:
        width = 2
    print("plotting channels...")
    model = Linear_regression(windowsize)
    model.load_state_dict(torch.load(MODEL_FOLDER + '/model.pt'))
    input = PIANO + VIOLIN
    start = len(input) // start
    input = input[start:start + width * windowsize]
    c0, c1 = runModel(model, input)
    p = PIANO[start + windowsize // 2:start + len(input) - windowsize // 2]
    v = VIOLIN[start + windowsize // 2:start + len(input) - windowsize // 2]
    subplot[0].plot(p)
    subplot[1].plot(v)
    subplot[2].plot(c0)
    subplot[3].plot(c1)
    print('done')

def plotValidatons(subplot, width=5, start=10):
    if start <= 0.9:
        start = 0.9
    if width <= 1:
        width = 2
    print('plotting validation loss...')
    with open(MODEL_FOLDER + "/times.npy", 'rb') as f:
        times = np.load(f)
    test_in = PIANO + VIOLIN
    start = len(test_in) // start
    models = len(times)
    dots = np.empty(models)
    mags = np.empty(models)
    for i in range(models):
        print('model', i, '/', models)
        model = Linear_regression(windowsize)
        model.load_state_dict(torch.load(MODEL_FOLDER + '/intermediate/model' + str(i) + '.pt'))
        if i == 0:
            test_in = test_in[start:start + width * windowsize]
        c0, c1 = runModel(model, test_in)
        mag0, mag1 = np.sqrt(sum(c0 ** 2)), np.sqrt(sum(c1 ** 2))
        comp_piano = PIANO[start + windowsize // 2:start + len(test_in) - windowsize // 2]
        piano_mag = np.sqrt(sum(comp_piano ** 2))
        comp_violin = VIOLIN[start + windowsize // 2:start + len(test_in) - windowsize // 2]
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

def DOTvsMSE(plot):
    with open(MODEL_FOLDER + '/losses.npy', 'rb') as f:
        losses = np.load(f)
    MSE_DOT = losses[:, 1:4:2] #select index 1 and 3 (MSE and DOT) with 1:4:2
    MSE_DOT = np.sort(MSE_DOT, axis=0)
    MSE = MSE_DOT[:, 0]
    DOT = MSE_DOT[:, 1]
    plot.set_xlabel("MSE")
    plot.set_ylabel("DOT")
    plot.plot(MSE, DOT)

def makeWavs(start=10, width=200):
    if start <= 0.9:
        start = 0.9
    width *= windowsize
    print("making wavs")
    model = Linear_regression(windowsize)
    model.load_state_dict(torch.load(MODEL_FOLDER + '/model.pt'))
    input = VIOLIN + PIANO
    start = len(input) // start
    input = input[start:start + width]
    c0, c1 = runModel(model, input)
    c0 *= 65536 / ampl
    c1 *= 65536 / ampl

    p = PIANO[start + windowsize // 2:start + width - windowsize // 2] * 65536 / ampl
    v = VIOLIN[start + windowsize // 2:start + width - windowsize // 2] * 65536 / ampl
    used_input = input[(windowsize // 2):width - (windowsize // 2)] * 65536 / ampl

    write("out/piano.wav", rate=44100, data=p.astype(np.int16))
    write("out/violin.wav", rate=44100, data=v.astype(np.int16))
    write("out/sum.wav", rate=44100, data=used_input.astype(np.int16))

    write("out/chanel0.wav", rate=44100, data=c0.astype(np.int16))
    write("out/chanel1.wav", rate=44100, data=c1.astype(np.int16))


#plot should be an array of size 3
def alphaSpeed(plot, alpha=0.1, width=2, start=10):
    if start <= 0.9:
        start = 0.9
    if width <= 1:
        width = 2
    input = VIOLIN + PIANO
    start = len(input) // start
    width *= windowsize
    input = input[start:start + width]
    running_mean = 0
    running_var = 0
    mean = np.empty(width)
    var = np.empty(width)
    for i in range(width):
        running_mean = (1 - alpha) * running_mean + alpha * input[i]
        running_var = (1 - alpha) * running_var + alpha * (input[i] - running_mean) ** 2
        mean[i] = running_mean
        var[i] = running_var
    #mean and VAR
    plot[0].plot(mean)
    plot[0].plot(var)
    plot[0].plot(np.ones(width) * sum((input - np.mean(input)) ** 2) / width)
    plot[0].plot(input)
    plot[0].legend(['running mean', 'running var', 'true var', 'input'])
    #DOT and mag
    running_dot = 0
    running_mag = np.zeros(2)
    vio = VIOLIN[start:start+width]
    pia = PIANO[start:start+width]
    DOT = np.empty(width)
    MAG = np.empty(width)
    DOT_norm = np.empty(width)
    for i in range(width):
        running_mag[0] = running_mag[0] * (1 - alpha) + alpha * (pia[i] ** 2)
        running_mag[1] = running_mag[1] * (1 - alpha) + alpha * (vio[i] ** 2)
        running_dot = running_dot * (1 - alpha) + alpha * pia[i] * vio[i]
        DOT[i] = running_dot ** 2 * (width ** 2)
        MAG[i] = running_mag[0] * running_mag[1] * (width ** 2)
        DOT_norm[i] = DOT[i] / MAG[i]
    true_DOT = sum(vio * pia) ** 2
    true_mag = sum(vio ** 2) * sum(pia ** 2)
    ax2 = plot[1].twinx()
    plot[1].set_ylabel("dots", color='blue')
    plot[1].plot(DOT, 'blue')
    plot[1].plot(true_DOT*np.ones(width), 'blue')
    ax2.set_ylabel("mags", color='red')
    ax2.plot(true_mag*np.ones(width), 'red')
    ax2.plot(MAG, 'red')
    plot[2].plot(DOT_norm)
    plot[2].plot(true_DOT / true_mag * np.ones(width))

def scoreAlpha(alpha=0.01, rep=3, width=2):
    if width <= 1:
        width = 2
    width = int(width * windowsize)
    VAR = 0
    DOT = 0
    for _ in range(rep):
        start = np.random.randint(0, len(VIOLIN) - width)
        vio = VIOLIN[start:start + width]
        start = np.random.randint(0, len(PIANO) - width)
        pia = PIANO[start:start + width]
        input = vio + pia
        true_var = sum((input - np.mean(input)) ** 2) / width
        m = sum(pia ** 2) * sum(vio ** 2)
        if m == 0:
            m = 1
        true_dot = sum(pia * vio) ** 2 / m
        running_mean = 0
        running_var = 0
        running_dot = 0
        running_mag = np.zeros(2) + 1e-6
        for i in range(width):
            running_mean = (1 - alpha) * running_mean + alpha * input[i]
            running_var = (1 - alpha) * running_var + alpha * ((input[i] - running_mean) ** 2)
            running_mag[0] = running_mag[0] * (1 - alpha) + alpha * (pia[i] ** 2)
            running_mag[1] = running_mag[1] * (1 - alpha) + alpha * (vio[i] ** 2)
            running_dot = (1 - alpha) * running_dot + alpha * pia[i] * vio[i]
            VAR += (running_var - true_var) ** 2
            DOT += (running_dot ** 2 / (running_mag[0] * running_mag[1]) - true_dot) ** 2
    return VAR / (rep * width), DOT / (rep * width)

def plotAlphaScores(plot, rep: int =128, r: int =32, width: int | float =2):
    VARS = []
    DOTS = []
    #use range between [0.0001, 0.5], a * (b ** i), (0, 0.5), (r, 2^-10 = 0.0001), gives:
    b = 2 ** (-9 / r)
    r = [0.5 * b ** i for i in range(r)]
    for alpha in r:
        v, d = scoreAlpha(alpha, rep, width)
        VARS.append(v)
        DOTS.append(d)
    plot.xscale('log')
    ax2 = plot.twinx()
    plot.ylabel("vars", color='blue')
    plot.plot(r, VARS, 'blue')
    ax2.set_ylabel("dots", color='red')
    ax2.plot(r, DOTS, 'red')

def plotWidthScore(plot, rep: int =128, r: int =32, alpha: float =0.01, domain=(1, 15)):
    s, e = domain
    if s < 1:
        s = 1
    e -= s
    #putting r in the range (but never truely 1)
    r = [s + e * (i / r) for i in range(1, 1+r)]
    VARS = []
    DOTS = []
    for width in r:
        v, d = scoreAlpha(alpha, rep, width)
        VARS.append(v * (width - windowsize))
        DOTS.append(d * (width - windowsize))
    ax2 = plot.twinx()
    plot.ylabel("vars", color='blue')
    plot.plot(r, VARS, 'blue')
    ax2.set_ylabel("dots", color='red')
    ax2.plot(r, DOTS, 'red')

def testDOT(rep=8, width=8):
    if width <= 1:
        width = 2
    width *= windowsize
    model = Linear_regression(windowsize)
    model.load_state_dict(torch.load(MODEL_FOLDER + '/model.pt'))
    DOT = 0
    for i in range(rep):
        print('-', i, '/', rep)
        start = np.random.randint(0, len(PIANO))
        pia = PIANO[start:start + width]
        start = np.random.randint(0, len(VIOLIN))
        vio = VIOLIN[start:start + width]
        input = pia + vio
        c0, c1 = runModel(model, input)
        DOT += sum(c0 * c1) ** 2 / (sum(c0 ** 2) * sum(c1 ** 2))
    return DOT / rep

#                               finding the best width factor (~8, not super exact, 4 and up is fine)
'''
plotWidthScore(plt, 512, 64)
plt.show()
'''

#                               finding the best alpha (0.01)
'''
plotAlphaScores(plt)
plt.show()
'''

#                               plotting how the running averages change based on alpha
'''
fig, ax = plt.subplots(3, 1)
alphaSpeed(ax, 0.01, 2)
plt.show()
'''

'''
print(testDOT())
input = VIOLIN + PIANO
start = len(input) // 10
input = input[start:start + len(input) // 50]
model = Linear_regression(windowsize)
model.load_state_dict(torch.load(MODEL_FOLDER + '/model.pt'))
c0, c1 = runModel(model, input)
fig, ax = plt.subplots(2, 1)
DOT = sum(c0 * c1) ** 2 / (sum(c0 ** 2) * sum(c1 ** 2))
print(DOT)
ax[0].plot(c0[:1000])
ax[1].plot(c1[:1000])
plt.show()
'''
print(testDOT(32))
#makeWavs(20)

'''
fig, ax = plt.subplots(2, 3)
plotLosses(ax[0][0])
#DOTvsMSE(ax[0][2])
plotChannels([ax[0][1], ax[0][2], ax[1][1], ax[1][2]])
#plotValidatons(ax[1][0], 2)
plt.show()
'''