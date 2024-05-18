import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from main import Model

def arrToDict(arr):
    out = {}
    for e in arr:
        if e[2] == 'int':
            out[e[0]] = int(e[1])
        elif e[2] == 'float':
            out[e[0]] = float(e[1])
    return out

print('plotting...')

with open('models/data5.npy', 'rb') as f:
    weights = np.load(f)
    losses = np.load(f)
    params = np.load(f)

params = arrToDict(params)
model = Model(windowsize=params['windowsize'], alpha=params['alpha'], lr=params['lr'],
            dotWeight=params['dotWeight'], b1=params['b1'], b2=params['b2'])

model.weights = weights

ampl = params['ampl']
with open('music/music.npy', 'rb') as f:
    PIANO = np.load(f)
    VIOLIN = np.load(f)

def makeWavs(start=10, width=100):
    print("making wavs")

    out = model.forward(start, start, width)

    c0 = out[0] * 65536
    c1 = out[1] * 65536

    p = PIANO[start + model.windowsize // 2:start + width - model.windowsize // 2] * 65536
    v = VIOLIN[start + model.windowsize // 2:start + width - model.windowsize // 2] * 65536

    used_input = p + v

    write("out/piano.wav", rate=8192, data=p.astype(np.int16))
    write("out/violin.wav", rate=8192, data=v.astype(np.int16))
    write("out/sum.wav", rate=8192, data=used_input.astype(np.int16))

    write("out/chanel0.wav", rate=8192, data=c0.astype(np.int16))
    write("out/chanel1.wav", rate=8192, data=c1.astype(np.int16))

#plot should be an array of size 3
def alphaSpeed(plot, alpha=0.1, width=2, start=10):
    if start <= 0.9:
        start = 0.9
    if width <= 1:
        width = 2
    input = VIOLIN + PIANO
    start = len(input) // start
    width *= model.windowsize
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
    plot[2].legend(['running dot', 'true dot'])

def scoreAlpha(alpha=0.01, rep=3, width=2024):
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

def plotAlphaScores(plot, rep: int =128, r: int =32, width: int | float =2024):
    VARS = []
    DOTS = []
    #use range between [0.0001, 0.5], a * (b ** i), (0, 0.5), (r, 2^-10 = 0.0001), gives:
    b = 2 ** (-12 / r)
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

def running_average(data, window=200):
    out = np.zeros(len(data) - window + 1)
    for i in range(len(out)):
        for j in range(window):
            out[i] += data[i + j]
    return out / window

def plotMeanlosses(window=1):
    fig, axs = plt.subplots(1, 2)
    varlosses = np.empty(len(losses))
    dotlosses = np.empty(len(losses))
    for i in range(len(losses)):
        dotlosses[i] = sum(losses[i][0]) / len(losses[i][0])
        varlosses[i] = sum(losses[i][1]) / len(losses[i][1])
    fig.supxlabel('episodes')
    axs[0].set_ylabel('var loss')
    axs[0].set_ylim((-1, 0))
    axs[1].set_ylabel('dot loss')
    axs[1].set_ylim((0, 1))
    axs[0].plot(running_average(varlosses, window))
    axs[1].plot(running_average(dotlosses, window))
    plt.show()

def plotChanels(startv=0, startp=0, width = 2024):
    fig, axs = plt.subplots(2, 3)
    out = model.forward(startp, startv, width)
    axs[0][0].plot(out[0])
    axs[0][1].plot(out[1])
    axs[0][2].plot(out[0] + out[1])
    startp = startp + model.windowsize // 2
    startv = startv + model.windowsize // 2
    width -= model.windowsize
    axs[1][0].plot(PIANO[startp:startp + width])
    axs[1][1].plot(VIOLIN[startv:startv + width])
    axs[1][2].plot(VIOLIN[startv:startv + width] + PIANO[startp:startp + width])
    plt.show()

#plotAlphaScores(plt, 400, 75, params[6])
#plotMeanlosses(200)

makeWavs(8000, 100000)

#plotChanels(startv=8000, startp=8000, width=128*2)