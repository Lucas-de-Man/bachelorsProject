from grengar import Grengar
import numpy as np
import time
import pickle

def startEnd(bars, valProp=0.1):
    global nrBars
    #np.ceil(nrBars * (1 - valProp)) makes sure we leave out enough for validation
    start = np.random.randint(0, np.ceil(nrBars * (1 - valProp)) - bars)
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

#model = Grengar(windowsize=128, regSize=32, alphaMain=0.00000000001, alphaReg=0.00001, energyMult=50)
with open("grengars/model-128-1500.obj", 'rb') as f:
    model = pickle.load(f)

#save start time
startTime = time.time()

steps = 1500
for i in range(steps):
    if (i % 10 == 1):
        reglosses, energy = model.losses()
        print("step", i, "of", steps, "reg0:", np.mean(reglosses[0][-50:]), "reg1:", np.mean(reglosses[1][-50:]),
                                               "energy:", np.mean(energy[-50:]))
        model.verbose = True
    #batchsize
    start, end = startEnd(8)
    model.batch(combination[0:barsize])
    model.verbose = False

#computed elapsed time
elapsed = time.time() - startTime
print("took", elapsed // 60, "minutes and", elapsed % 60, "seconds to train.")

model.save()
