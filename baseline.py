import numpy as np
import os
import time

with open('music/music.npy', 'rb') as f:
    piano = np.load(f)
    violin = np.load(f)
    barsize = np.load(f)

class linReg:
    def __init__(self, windowsize=1024, stepsize=len(piano)):
        self.windowsize = windowsize
        self.stepsize = stepsize - windowsize
        self.solution = np.empty(2*windowsize + 1)
        self.xxSum = np.zeros((2*windowsize + 1, 2*windowsize + 1))
        self.xySum = np.zeros((1, 2*windowsize + 1))

    def addStep(self, start, length):
        global piano
        global violin
        targetMat = piano[start + self.windowsize // 2 : start + length + self.windowsize // 2]
        inputMat = np.empty((length, 2*self.windowsize + 1))
        #first line of matrix set
        inputMat[0][2*self.windowsize] = 1
        for i in range(self.windowsize):
            inputMat[0][i] = piano[start + i] + violin[start + i]
            inputMat[0][i + self.windowsize] = inputMat[0][i] * inputMat[0][i]

        for i in range(1, length):
            #bias
            inputMat[i][2 * self.windowsize] = 1
            #shift values
            for j in range(self.windowsize - 1):
                inputMat[i][j] = inputMat[i - 1][j + 1]
                inputMat[i][j + self.windowsize] = inputMat[i - 1][j + self.windowsize + 1]
            #set the new values in
            inputMat[i][self.windowsize - 1] = piano[start + self.windowsize + i] + violin[start + self.windowsize + i]
            inputMat[i][2*self.windowsize - 1] = inputMat[i][self.windowsize - 1] * inputMat[i][self.windowsize - 1]

        self.xxSum += np.dot(inputMat.transpose(), inputMat)
        self.xySum += np.dot(inputMat.transpose(), targetMat)

    def add(self, start = 0, length=len(piano)):
        length -= self.windowsize
        i = start
        for i in range(start, length - self.stepsize, self.stepsize):
            self.addStep(i, self.stepsize)
        self.addStep(i, length - i)

    def solve(self):
        self.solution = np.dot(self.xySum, np.linalg.inv(self.xxSum))
        return self.solution


regressor = linReg(windowsize=128)

start = time.time()

regressor.add()
weights = regressor.solve()

print("minutes elapsed", (time.time() - start) / 60)


#saving
def arrToDict(arr):
    out = {}
    for e in arr:
        if e[2] == 'int':
            out[e[0]] = int(e[1])
        elif e[2] == 'float':
            out[e[0]] = float(e[1])
    return out

folder = 'models/baseline'
if not os.path.exists(folder):
    os.makedirs(folder)

modelNr = 0
if os.path.exists(folder + '/modelNr.txt'):
    with open(folder + '/modelNr.txt', 'r') as f:
        modelNr = int(f.read())

with open(folder + "/baseline" + str(modelNr) + '.npy', 'wb') as f:
    np.save(f, weights[0])
    np.save(f, regressor.windowsize)

with open(folder + '/modelNr.txt', 'w') as f:
    f.write(str(modelNr + 1))