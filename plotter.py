import numpy as np
import matplotlib.pyplot as plt
import wave
from main import Model
import os

def arrToDict(arr):
    out = {}
    for e in arr:
        if e[2] == 'int':
            out[e[0]] = int(e[1])
        elif e[2] == 'float':
            out[e[0]] = float(e[1])
    return out

print('plotting...')

with open('models/baseline/baseline4.npy', 'rb') as f:
    weights = np.load(f)
    windowsize = np.load(f)
    #losses = np.load(f)
    #params = np.load(f)

print(windowsize)

#params = arrToDict(params)
#model = Model(windowsize=params['windowsize'], alpha=params['alpha'], lr=params['lr'],
#            dotWeight=params['dotWeight'], b1=params['b1'], b2=params['b2'])
model = Model(windowsize=windowsize)
model.weights = weights

with open('music/music.npy', 'rb') as f:
    PIANO = np.load(f)
    VIOLIN = np.load(f)
    barsize = np.load(f)



def makeWavs(start=10, width=100):
    print("making wavs")

    if not os.path.exists('out'):
        os.makedirs('out')

    out = model.forward(start, start, width)

    c0 = out[0]
    c1 = out[1]

    c0 -= min(c0)
    c1 -= min(c1)
    c0 *= 2147483647. / max(c0)
    c1 *= 2147483647. / max(c1)
    c0 = c0.astype(int)
    c1 = c1.astype(int)

    p = PIANO[start + model.windowsize // 2:start + width - model.windowsize // 2]
    v = VIOLIN[start + model.windowsize // 2:start + width - model.windowsize // 2]

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

def plotAlphaScores(plot, rep: int =128, r: int =32, width: int | float = 2024):
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

def plotChanels(start=0, width = 2024):
    fig, axs = plt.subplots(2, 3)
    out = model.forward(start, start, width)
    axs[0][0].plot(out[0])
    axs[0][1].plot(out[1])
    axs[0][2].plot(out[0] + out[1])
    startp = start + model.windowsize // 2
    startv = start + model.windowsize // 2
    width -= model.windowsize
    axs[1][0].plot(PIANO[startp:startp + width])
    axs[1][1].plot(VIOLIN[startv:startv + width])
    axs[1][2].plot(VIOLIN[startv:startv + width] + PIANO[startp:startp + width])
    plt.show()

def chanelsPlot(startv=0, startp=0, width=128):
    out = model.forward(startp, startv, width + model.windowsize)
    plt.plot(out[0], label='chanel 0')
    plt.plot(out[1], label='chanel 1')
    plt.legend()
    plt.title('chanel 0 and chanel 1')
    plt.xlabel('t')
    plt.show()

def dataDot(rep=100, length=4096):
    final = 0
    for _ in range(rep):
        startP = np.random.randint(0, (len(PIANO) - length) // barsize) * barsize
        startV = np.random.randint(0, (len(VIOLIN) - length) // barsize) * barsize
        dot = 0
        piaMag = 0
        vioMag = 0
        for i in range(length):
            dot += PIANO[startP+i] * VIOLIN[startV+i]
            piaMag += PIANO[startP+i] ** 2
            vioMag += VIOLIN[startV+i] ** 2
        final += dot * dot / (piaMag * vioMag)
    return final / rep

def scoreModel(rep=10, length=4096):
    score = 0
    for i in range(rep):
        startP = np.random.randint(0, (len(PIANO) - length - model.windowsize) // barsize) * barsize
        startV = np.random.randint(0, (len(VIOLIN) - length - model.windowsize) // barsize) * barsize
        out = model.forward(startP, startV, length + model.windowsize)
        outMag = [sum(out[0] ** 2), sum(out[1] ** 2)]
        pianoMag = sum(PIANO[startP:startP+length] ** 2)
        violinMag = sum(VIOLIN[startV:startV + length] ** 2)
        score0 = sum(out[0] * PIANO[startP:startP+length]) ** 2 / (outMag[0] * pianoMag)
        score0 += sum(out[1] * VIOLIN[startV:startV+length]) ** 2 / (outMag[1] * violinMag)
        score1 = sum(out[1] * PIANO[startP:startP + length]) ** 2 / (outMag[1] * pianoMag)
        score1 += sum(out[0] * VIOLIN[startV:startV + length]) ** 2 / (outMag[0] * violinMag)
        score += min(score0, score1)
        if i % (rep // 5) == 0:
            print(i, '/', rep)
    return score / rep

def intensity(data, alpha=0.05):
    intens = 0
    intensity = np.empty(len(data))
    for i in range(len(data)):
        intens = alpha*(data[i] ** 2) + (1-alpha) * intens
        intensity[i] = intens
    return intensity

def spectogram(data, sliceWidth=1000, plot=plt):
    specto = np.empty((len(data) - sliceWidth, sliceWidth // 2))
    subset = np.empty(sliceWidth)
    for i in range(len(specto)):
        for j in range(sliceWidth):
            subset[j] = data[i+j]
        specto[i] = np.abs(np.fft.fft(subset))[:sliceWidth // 2]
    specto = np.rot90(specto)
    plot.imshow(specto)
    for i in range(1, 1 + (len(data) - sliceWidth) // barsize):
        plot.vlines(i * barsize - sliceWidth, 0, sliceWidth // 2, colors='green')
        plot.vlines(i * barsize, 0, sliceWidth // 2, colors='red')
    if plot == plt:
        plot.show()

def slope(data, N=99):
    slopes = np.zeros(len(data))
    xy = 0
    sy = 0
    x = N*(N+1) // 2
    under = (N+1)*(N+2)*(2*N+3) // 6 * N - x * x
    for i in range(N):
        sy += data[i]
        xy += i * data[i]
    for i in range(N, len(data)):
        sy += data[i]
        xy += N * data[i]
        slopes[i - N // 2] = ((N+1) * xy - x*sy) / under
        sy -= data[i - N]
        xy -= sy
    return slopes

def intensityTest():
    width = 5000
    start = barsize * 17
    data = VIOLIN[start:start + width]
    data += PIANO[start:start + width]
    fig, axs = plt.subplots(2, 3)
    intens = intensity(data=data, alpha=0.025)
    axs[0][0].plot(data)
    axs[0][1].plot(intens)
    slope2 = slope(intens) ** 2
    axs[0][1].plot(slope2 * max(intens) / max(slope2))
    for i in range(1, 1 + width // barsize):
        axs[0][1].vlines(i * barsize, min(intens), max(intens), colors='red')

    spectogram(data, plot=axs[0][2], sliceWidth=200)

    spectogram(VIOLIN[start:start + width], plot=axs[1][0], sliceWidth=200)
    spectogram(PIANO[start:start + width], plot=axs[1][1], sliceWidth=200)

    plt.show()

#chanelsPlot()

#plotAlphaScores(plt, 400, 75, params[6])
#plotMeanlosses(50)

#intensityTest()

#spectogram(VIOLIN[0:barsize*8])

#print(scoreModel(rep=100))

#makeWavs(100, 100000)

#plotChanels(start=10000, width=1024 + model.windowsize)

#print(dataDot())

#model-data dot
#0.10238340859489785 (aligned)
#0.06364016318678055 (unaligned)

#original data dot
#0.0006275056974094232 (unaligned)
#0.009679177712664539 (aligned)