from grengar import Grengar
import numpy as np
import time
import pickle

def startEnd(data, size, valProp=0.1):
    #np.ceil(nrBars * (1 - valProp)) makes sure we leave out enough for validation
    start = np.random.randint(0, np.ceil(len(data) * (1 - valProp)) - size)
    return start, start + size

with open('music/music.npy', 'rb') as f:
    piano = np.load(f)
    violin = np.load(f)
    barsize = np.load(f)

combination = piano + violin
nrBars = len(piano) // barsize

#model = Grengar(windowsize=32, regSize=3, alphaMain=1e-6, alphaReg=1e-5, energyMult=10, energyAlpha=0.9999, beta1=0.999, beta2=0.9999, batchSaveRate=1000)
with open("grengars/model-32-401.obj", 'rb') as f:
    model = pickle.load(f)

'''
model.mainWeights = modelP.mainWeights
model.mainBias = modelP.mainBias
model.regWeights1 = modelP.regWeights1
model.regWeights0 = modelP.regWeights0
model.regBias0 = modelP.regBias0
model.regBias1 = modelP.regBias1
'''

print(model.alphaMain, model.energyMult)

model.energyMult = 1e12
model.alphaMain = 1e-9
model.alphaReg = 1e-8
model.energyAlpha = 0.9999

#save start time
startTime = time.time()

steps = 200000
printEvery = 5000
atRate = printEvery // model.saveRate
for i in range(steps):
    if i % printEvery == atRate:
        reglosses, energy = model.losses()
        mainGrad, energyGrad, regGrad = model.gradMags()
        print("step", i, "of", steps, "reg0:", np.mean(reglosses[0][-atRate:]), "reg1:", np.mean(reglosses[1][-atRate:]),
                                      "energy:", np.mean(energy[-atRate:]))
        print("mainGrad:", np.mean(mainGrad[-atRate:]), "regGrad:", np.mean(regGrad[-atRate:]),
                                                        "energyGrad:", np.mean(energyGrad[-atRate:]))
        print("-----------------")
        model.verbose = True
    #batchsize
    startP, endP = startEnd(piano, model.windowsize + 10)
    startV, endV = startEnd(violin, model.windowsize + 10)
    model.batch(piano[startP:endP] + violin[startV:endV])
    model.verbose = False

#computed elapsed time
elapsed = time.time() - startTime
print("took", elapsed // 60, "minutes and", elapsed % 60, "seconds to train.")

model.save()
