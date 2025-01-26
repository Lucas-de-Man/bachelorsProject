from grengar import Grengar
import numpy as np
import time

def startEnd(bars):
    global nrBars
    start = np.random.randint(0, nrBars - bars)
    return start * barsize, (start + bars) * barsize

with open('music/music.npy', 'rb') as f:
    piano = np.load(f)
    violin = np.load(f)
    barsize = np.load(f)

def loadModel(path):
    model = Grengar()
    with open(path, 'rb') as f:
        #model values
        model.windowsize = np.load(f)
        model.mainWeights = np.load(f)
        model.mainBias = np.load(f)
        model.alphaMain = np.load(f)
        #regression values
        model.regSize = np.load(f)
        model.regWeights0 = np.load(f)
        model.regWeights1 = np.load(f)
        model.regBias0 = np.load(f)
        model.regBias1 = np.load(f)
        model.alphaReg = np.load(f)

    return model

combination = piano + violin
nrBars = len(piano) // barsize

model = Grengar(windowsize=64, regSize=32, alphaMain=0.000001, magMult=10)
#model = loadModel("grengars/model-64-1000.npy")


#save start time
startTime = time.time()

steps = 7500
for i in range(steps):
    if (i % 100 == 0):
        print("step", i, "of", steps, "loss:")
        model.verbose = True
    #batchsize
    start, end = startEnd(8)
    model.step(combination[0:barsize])
    model.verbose = False

#computed elapsed time
elapsed = time.time() - startTime
print("took", elapsed // 60, "minutes and", elapsed % 60, "seconds to train.")

with open("grengars/model-" + str(model.windowsize) + '-' + str(steps) + ".npy", 'wb') as f:
    np.save(f, model.windowsize)
    np.save(f, model.mainWeights)
    np.save(f, model.mainBias)
    np.save(f, model.alphaMain)
    # regression
    np.save(f, model.regSize)
    np.save(f, model.regWeights0)
    np.save(f, model.regWeights1)
    np.save(f, model.regBias0)
    np.save(f, model.regBias1)
    np.save(f, model.alphaReg)
