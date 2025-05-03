from grengar import Grengar
import numpy as np
import matplotlib.pyplot as plt
import wave
import os
from regressionGrengar import solveLinearReg, Regression
import pickle
import os.path

#setting up the loaded model
with open('grengars/model-32-601.obj', 'rb') as f:
    model = pickle.load(f)

#loading the data
with open('music/music.npy', 'rb') as f:
    piano = np.load(f)
    violin = np.load(f)
    maxShift = np.load(f)

def plotChanels(piano, violin, multiplot=False):
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
    if not multiplot:
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
    pianoWeights, pianoBias = regPiano.solve()
    violinWeights, violinBias = regViolin.solve()
    pianoReg = Regression(pianoWeights, pianoBias)
    violinReg = Regression(violinWeights, violinBias)
    return pianoReg.mse(piano, violin) + violinReg.mse(violin, piano)

def plotLossWindowsize(windowsizes=[4], alphas=50, piano=piano, violin=violin, size=-1):
    if size < max(windowsizes):
        size = len(piano)
    for i, windowsize in enumerate(windowsizes):
        print("windowsize", i, "of", len(windowsizes))
        print("start windowsize =", windowsize)
        mses = []
        for alpha in range(alphas + 1):
            a = 0.5 + alpha / alphas / 2
            p = piano[:size] * a + violin[:size] * (1 - a)
            v = piano[:size] * (1 - a) + violin[:size] * a
            regPiano = solveLinearReg(windowsize)
            regViolin = solveLinearReg(windowsize)
            regPiano.addStep(p, v[:size])
            regViolin.addStep(v, p[:size])
            pianoWeights, pianoBias = regPiano.solve()
            violinWeights, violinBias = regViolin.solve()
            pianoReg = Regression(pianoWeights, pianoBias)
            violinReg = Regression(violinWeights, violinBias)
            mses.append(pianoReg.mse(p, v) + violinReg.mse(v, p))
        maxLoss = sum((piano[:size] - np.mean(piano[:size])) ** 2) + sum((violin[:size] - np.mean(violin[:size])) ** 2)
        mses = np.array(mses) * size / maxLoss
        plt.plot([0.5 + x / alphas / 2 for x in range(alphas + 1)], mses, label=str(windowsize))
    plt.xlabel('mix')
    plt.ylim((0, 1))
    plt.ylabel('MSE-loss')
    plt.legend()
    plt.show()

def runningMean(data, windowSize):
    return np.convolve(np.array(data), np.array([1 / windowSize for _ in range(windowSize)]), 'valid')

def expectedLosses(path, valProp=0.1, steps=100):
    with open(path, 'rb') as f:
        perfect = pickle.load(f)
    perfectReg0 = Regression(perfect.regWeights0, perfect.regBias0)
    perfectReg1 = Regression(perfect.regWeights1, perfect.regBias1)
    mse = 0
    reg0 = 0
    reg1 = 0
    for s in range(steps):
        summedData = np.empty(len(piano))
        #only use the validation part
        start = int(s * maxShift * (1 - valProp) / steps)#int((1 - valProp) * maxShift + s * maxShift * valProp / steps)
        for i in range(len(summedData)):
            summedData[i] = piano[i] + violin[(start + i) % len(violin)]
        channel0, channel1 = perfect.forward(summedData)
        dataEnergy = sum(summedData[perfect.windowsize // 2:-perfect.windowsize + perfect.windowsize // 2 + 1] ** 2) / 2
        mse += ((sum(channel0 ** 2) - dataEnergy) ** 2 + (sum(channel1 ** 2) - dataEnergy) ** 2) / (len(channel0) ** 2)
        reg0 += perfectReg0.mse(channel0, channel1)
        reg1 += perfectReg1.mse(channel1, channel0)
    return mse / steps, reg0 / steps, reg1 / steps

def plotLosses(skip=0, slidingWindow=1, multiplot=False, compareTo=''):
    global model
    regLosses, energyLosses = model.losses()
    x = [skip + i for i in range(len(regLosses[0]) - skip - slidingWindow + 1)]


    regLosses = [runningMean(regLosses[0][skip:], slidingWindow), runningMean(regLosses[1][skip:], slidingWindow)]
    energyLosses = runningMean(energyLosses[skip:], slidingWindow)
    regLoss = regLosses[0] + regLosses[1]

    fig, ax = plt.subplots()

    ax.set_xlabel('step')
    ax.set_ylabel('summed energy MSE', color='green')
    ax.set_yscale('log')

    line1 = ax.plot(x, energyLosses, color='green', label='summed energy MSE')
    ax2 = ax.twinx()
    line2 = ax2.plot(x, regLoss, color='red', label='Granger losses')
    ax2.set_ylabel('negative Granger loss', color='red')
    ax2.set_yscale('log')

    if os.path.exists(compareTo) and os.path.isfile(compareTo):
        energy, reg0, reg1 = expectedLosses(compareTo)
        ax.hlines(energy, 0, len(energyLosses), color='green')
        ax2.hlines(reg0 + reg1, 0, len(energyLosses), color='red')
        print(energy, reg0, reg1)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc=0)
    if not multiplot:
        plt.show()

#pianoPredicted, violinPredicted = model.forward(piano + violin)
#plotLossWindowsize(windowsizes=[3, 4, 5, 6], piano=pianoPredicted, violin=violinPredicted, size=-1)

plotChanels(piano[0:3*1024], violin[0:3*1024], True)

#plotChanels(piano[0:2*1024], violin[int(0.95 * maxShift):int(0.95 * maxShift) + 2*1024])

plotLosses(0, 5, True)#, 'grengars/perfect/model-32-128-10.obj')

plt.show()

#makeWavs(piano[0:20*maxShift], violin[0:20*maxShift])
'''
predictedPiano, predictedViolin = model.forward(piano + violin)
piaLoss = sum((predictedPiano - piano[model.windowsize // 2:-model.windowsize + model.windowsize // 2 + 1]) ** 2)
vioLoss = sum((predictedViolin - violin[model.windowsize // 2:-model.windowsize + model.windowsize // 2 + 1]) ** 2)
print((piaLoss + vioLoss) / len(predictedViolin) / 2)
'''