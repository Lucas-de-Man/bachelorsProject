from grengar import Grengar
import numpy as np
import matplotlib.pyplot as plt
import wave
import os
from regressionGrengar import solveLinearReg, Regression
import pickle

#setting up the loaded model
with open('grengars/model-128-1500.obj', 'rb') as f:
    model = pickle.load(f)

#loading the data
with open('music/music.npy', 'rb') as f:
    piano = np.load(f)
    violin = np.load(f)
    barsize = np.load(f)

def plotChanels(piano, violin):
    global model
    fig, axs = plt.subplots(2, 3)
    together = violin + piano
    c0, c1 = model.forward(together)
    axs[0][0].plot(c0)
    axs[0][1].plot(c1)
    axs[0][2].plot(c0 + c1)
    axs[1][0].plot(piano[model.windowsize // 2:-model.windowsize // 2])
    axs[1][1].plot(violin[model.windowsize // 2:-model.windowsize // 2])
    axs[1][2].plot(together[model.windowsize // 2:-model.windowsize // 2])
    plt.show()

def makeWavs(piano, violin):
    print("making wavs")

    if not os.path.exists('out'):
        os.makedirs('out')

    c0, c1 = model.forward(piano + violin)

    c0 -= min(c0)
    c1 -= min(c1)
    c0 *= 2147483647. / max(c0)
    c1 *= 2147483647. / max(c1)
    c0 = c0.astype(int)
    c1 = c1.astype(int)

    p = piano[model.windowsize // 2:-model.windowsize + model.windowsize // 2]
    v = violin[model.windowsize // 2:-model.windowsize + model.windowsize // 2]

    used_input = p + v

    used_input -= min(used_input)
    used_input *= 2147483647. / max(used_input)
    used_input = used_input.astype(int)

    p -= min(p)
    v -= min(v)
    p *= 2147483647. / max(p)
    v *= 2147483647. / max(v)
    p = p.astype(int)
    v = v.astype(int)

    with wave.open("out/piano.wav", mode='wb') as f:
        f.setnchannels(1)
        f.setsampwidth(4)
        # 4096, 128 = 65.4 Hz, so 128/65.4=1.957 sec. 4096/1.957=2093 frames/sec
        f.setframerate(8192)
        f.writeframes(bytes(p))
    with wave.open("out/violin.wav", mode='wb') as f:
        f.setnchannels(1)
        f.setsampwidth(4)
        # 4096, 128 = 65.4 Hz, so 128/65.4=1.957 sec. 4096/1.957=2093 frames/sec
        f.setframerate(8192)
        f.writeframes(bytes(v))
    with wave.open("out/sum.wav", mode='wb') as f:
        f.setnchannels(1)
        f.setsampwidth(4)
        # 4096, 128 = 65.4 Hz, so 128/65.4=1.957 sec. 4096/1.957=2093 frames/sec
        f.setframerate(8192)
        f.writeframes(bytes(used_input))
    with wave.open("out/chanel0.wav", mode='wb') as f:
        f.setnchannels(1)
        f.setsampwidth(4)
        # 4096, 128 = 65.4 Hz, so 128/65.4=1.957 sec. 4096/1.957=2093 frames/sec
        f.setframerate(8192)
        f.writeframes(bytes(c0))
    with wave.open("out/chanel1.wav", mode='wb') as f:
        f.setnchannels(1)
        f.setsampwidth(4)
        # 4096, 128 = 65.4 Hz, so 128/65.4=1.957 sec. 4096/1.957=2093 frames/sec
        f.setframerate(8192)
        f.writeframes(bytes(c1))

    print("made wavs")

def bestLoss(piano, violin, windowsize):
    regPiano = solveLinearReg(windowsize)
    regViolin = solveLinearReg(windowsize)
    regPiano.addStep(piano, violin)
    regViolin.addStep(violin, piano)
    pianoWeights = regPiano.solve()[0]
    violinWeights = regViolin.solve()[0]
    pianoReg = Regression(pianoWeights[:-1], pianoWeights[-1])
    violinReg = Regression(violinWeights[:-1], violinWeights[-1])
    return pianoReg.mse(piano, violin) + violinReg.mse(violin, piano)

def plotLossWindowsize(windowsizes=[8, 16, 32, 64], alpha=0.9):
    losses = []
    pComb = alpha * piano + (1 - alpha) * violin
    vComb = alpha * violin + (1 - alpha) * piano
    for ws in windowsizes:
        losses.append(bestLoss(pComb, vComb, ws))
    plt.xscale('log')
    plt.ylim((0, 0.5))
    plt.plot(windowsizes, losses)
    plt.xlabel('windowsize')
    plt.ylabel('MSE-loss')
    plt.show()

def plotLosses(skip=500):
    global model
    regLosses, orthLosses = model.losses()
    x = [skip + i for i in range(len(regLosses[0]) - skip)]

    regLosses = [np.array(regLosses[0][skip:]), np.array(regLosses[1][skip:])]
    orthLosses = np.array(orthLosses[skip:])
    total = -regLosses[0] * regLosses[1] + orthLosses

    fig, ax = plt.subplots()

    ax.set_xlabel('step')
    ax.set_ylabel('loss')

    ax.plot(x, regLosses[0], color='blue', label='regression c0')
    ax.plot(x, regLosses[1], color='purple', label='regression c1')
    #ax.plot(x, orthLosses, color='green', label='orthogonality')
    #ax.plot(x, total, color='red', label='total')

    ax.legend()
    plt.show()

#plotLossWindowsize([8, 16, 32, 64], 0.95)

#plotChanels(piano[0:256], violin[0:256])

#plotLosses(0)

#makeWavs(piano[0:2*barsize], violin[0:2*barsize])