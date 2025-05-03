import numpy as np
from regressionGrengar import solveLinearReg, solveQuadraticRegression, Regression
from grengar import Grengar
import pickle

with open('music/music.npy', 'rb') as f:
    piano = np.load(f)
    violin = np.load(f)
    maxShift = np.load(f)

model = Grengar(windowsize=32, regSize=3)

mainSolver = solveQuadraticRegression(model.windowsize)
regSolverPV = solveLinearReg(model.regSize)
regSolverVP = solveLinearReg(model.regSize)

steps = 128
valProp = 0.1 #10% validation data (the end of the possible shifts are used for validation)
percAlong = 10
for s in range(steps):
    if s == steps * percAlong // 100:
        print(s * 100 / steps, "% done")
        percAlong += 10
    summedData = np.empty(len(piano))
    start = int(s * maxShift * (1 - valProp) / steps)
    for i in range(len(summedData)):
        #fill summed data with the shifted sum, looping piano around
        summedData[i] = piano[i] + violin[(start + i) % len(violin)]
    mainSolver.addStep(summedData, piano)

print("100 % done")
#find and update perfect weights
mainWeights, mainBias = mainSolver.solve()
model.mainWeights = mainWeights
model.mainBias = mainBias
print("main done")
percAlong = 10
for s in range(steps):
    if s == steps * percAlong // 100:
        print(s * 100 / steps, "% done")
        percAlong += 10
    summedData = np.empty(len(piano))
    start = int(s * maxShift * (1 - valProp) / steps)
    for i in range(len(summedData)):
        # fill summed data with the shifted sum, looping piano around
        summedData[i] = piano[i] + violin[(start + i) % len(violin)]
    #recreate targets
    predictedPiano, predictedViolin = model.forward(summedData)
    #train regression model 0
    regSolverPV.addStep(predictedPiano, predictedViolin)
    regWeights, regBias = regSolverPV.solve()
    model.regWeights0 = regWeights[::-1]
    model.regBias0 = regBias
    #train regression model 1
    regSolverVP.addStep(predictedViolin, predictedPiano)
    regWeights, regBias = regSolverVP.solve()
    model.regWeights1 = regWeights[::-1]
    model.regBias1 = regBias
print("reg done")

path = "grengars/perfect/model-" + str(model.windowsize) + "-" + str(steps) + "-" + str(int(100 * valProp)) + ".obj"
with open(path, 'wb') as f:
    pickle.dump(model, f)