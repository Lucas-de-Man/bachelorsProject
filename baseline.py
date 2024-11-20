import numpy as np
import os
import time


with open('music/music.npy', 'rb') as f:
    piano = np.load(f)
    violin = np.load(f)
    barsize = np.load(f)

class linReg:
    def __init__(self, windowsize=1204, stepsize=len(piano)):
        self.windowsize=windowsize
        self.stepsize = stepsize - windowsize
        self.solution = np.empty(2*windowsize + 1)
        self.xxSum = np.zeros((2*windowsize + 1, 2*windowsize + 1))
        self.xySum = np.zeros((1, 2*windowsize + 1))

    def addStep(self, start, length):
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

    def add(self, start=0, length=len(piano)):
        length -= self.windowsize
        i=start
        for i in range(start, length - self.stepsize, self.stepsize):
            self.addStep(i, self.stepsize)
        self.addStep(i, length - i)

    def solve(self):
        self.solution = np.dot(self.xySum, np.linalg.inv(self.xxSum))
        return self.solution;


class linearRegression:
    def __init__(self, windowsize=256):
        self.windowsize = windowsize
        self.xMatrix = np.zeros((2*windowsize + 1, 2*windowsize + 1))
        self.xy = np.zeros((1, 2*windowsize + 1))

    def add(self, startP, startV, size):
        size += self.windowsize
        subX = np.zeros((2*self.windowsize + 1, 1))
        subXT = np.zeros((1, 2*self.windowsize + 1))
            #initialize sub vectors
        #biases
        subX[2*self.windowsize][0] = 1
        subXT[0][2*self.windowsize] = 1
        for i in range(self.windowsize):
            subX[i][0] = piano[startP + i] + violin[startV + i]
            #square
            subX[i + self.windowsize][0] = subX[i][0] * subX[i][0]
            #transposed
            subXT[0][i] = subX[i][0]
            subXT[0][i + self.windowsize] = subX[i + self.windowsize][0]
        for start in range(self.windowsize, size):
            if (start - self.windowsize) % 10 == 0:
                print((start - self.windowsize), '/',  size - self.windowsize)
            #add to the matrix and vector sum we need
            for i in range(self.windowsize):
                curI = (start + i) % self.windowsize
                self.xMatrix[i][2*self.windowsize] += subXT[0][curI]
                self.xMatrix[i + self.windowsize][2 * self.windowsize] += subXT[0][curI + self.windowsize]
                self.xy[0][i] += subXT[0][curI] * piano[startP + start - self.windowsize + self.windowsize // 2]
                self.xy[0][i + self.windowsize] += subXT[0][curI + self.windowsize] * piano[startP + start - self.windowsize + self.windowsize // 2]
                for j in range(self.windowsize):
                    curJ = (start + j) % self.windowsize
                    self.xMatrix[i][j] += subXT[0][curI] * subX[curJ][0]
                    self.xMatrix[i + self.windowsize][j] += subXT[0][curI + self.windowsize] * subX[curJ][0]
                    self.xMatrix[i][j + self.windowsize] += subXT[0][curI] * subX[curJ + self.windowsize][0]
                    self.xMatrix[i + self.windowsize][j + self.windowsize] += subXT[0][curI + self.windowsize] * subX[curJ + self.windowsize][0]
            for j in range(self.windowsize):
                curJ = (start + j) % self.windowsize
                self.xMatrix[2*self.windowsize][j] += subX[curJ][0]
                self.xMatrix[2 * self.windowsize][j + self.windowsize] += subX[curJ + self.windowsize][0]
            #biases
            self.xMatrix[2*self.windowsize][2*self.windowsize] += 1
            self.xy[0][2*self.windowsize] += piano[startP + start - self.windowsize + self.windowsize // 2]
            #overwrite oldest value with the next
            update = start % self.windowsize
            subX[update][0] = piano[startP + start] + violin[startV + start]
            subX[update + self.windowsize][0] = subX[update][0] * subX[update][0]
            subXT[0][update] = subX[update][0]
            subXT[0][update + self.windowsize] = subX[update + self.windowsize][0]


    def compWeights(self):
        print("matrix")
        #print(self.xMatrix)
        #print("xy")
        #print(self.xy)
        inv = np.linalg.inv(self.xMatrix)
        print("inverse")
        #print(inv)
        return np.dot(self.xy, inv)

regressor = linReg(windowsize=512)

start = time.time()

regressor.add()
weights = regressor.solve()

print("minutes elapsed", (time.time() - start) / 60)

print("weights")
print(weights)


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