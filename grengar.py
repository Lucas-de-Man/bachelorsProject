import numpy as np
import pickle

class Grengar:
    def __init__(self, windowsize=128, regSize=32, alphaMain=0.000001, energyMult=10, alphaReg=0.001, energyAlpha=0.99, verbose=False):
        self.windowsize = windowsize
        self.regSize = regSize
        self.alphaMain = alphaMain
        self.energyMult = energyMult
        self.alphaReg = alphaReg
        self.verbose = verbose
        self.energyAlpha = energyAlpha

            #weigts
        #main weights have linear and quadratic components and a bias
        self.mainWeights = [np.random.normal(0, 1 / windowsize, windowsize) for _ in range(2)]
        self.mainBias = np.random.normal(0, 1 / windowsize)
        #kept in reverse for convolutions
        self.regWeights0 = np.random.normal(0, 1 / regSize, regSize)
        self.regBias0 = np.random.normal(0, 1 / regSize)
        self.regWeights1 = np.random.normal(0, 15 / regSize, regSize)
        self.regBias1 = np.random.normal(0, 1 / regSize)

            #equal energy grad
        self.chanelEneries = [0, 0]
        self.inputEnergy = 0
        self.energyGrads0 = [np.zeros(windowsize) for _ in range(2)]
        self.energyGrads1 = [np.zeros(windowsize) for _ in range(2)]
        self.energyGradBias = [0, 0]

        #keep track
        self.reglosses = [[], []]
        self.energyLoss = []

    def save(self, path=""):
        if path == "":
            path = "grengars/model-" + str(self.windowsize) + '-' + str(len(self.energyLoss)) + ".obj"
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def losses(self):
        return self.reglosses, self.energyLoss

    def forward(self, input):
        if len(input) < self.windowsize:
            print("input needs to be bigger than or equal to the windwsize")
            return
        #linear terms
        chanel0  = np.convolve(self.mainWeights[0], input, 'valid')
        #quadratic terms
        chanel0 += np.convolve(self.mainWeights[1], input ** 2, 'valid')
        #bias
        chanel0 += self.mainBias
        chanel1 = input[self.windowsize // 2:-self.windowsize + self.windowsize // 2 + 1] - chanel0
        return chanel0, chanel1

    def batch(self, input):
        rl0 = []
        rl1 = []
        el  = []
        for i in range(len(input) - self.windowsize - self.regSize + 1):
            regloss0, regloss1, energyloss = self.step(input[i:i + self.windowsize + self.regSize - 1])
            rl0.append(regloss0)
            rl1.append(regloss1)
            el.append(energyloss)
        self.reglosses[0].append(np.mean(rl0))
        self.reglosses[1].append(np.mean(rl1))
        self.energyLoss.append(np.mean(el))
        self.reset()

    def reset(self):
        self.chanelEneries = [0, 0]
        self.inputEnergy = 0
        self.energyGrads0 = [np.zeros(self.windowsize) for _ in range(2)]
        self.energyGrads1 = [np.zeros(self.windowsize) for _ in range(2)]
        self.energyGradBias = [0, 0]

    #x being length windowsize + regSize - 1
    def step(self, x):
        x2 = x * x
        c0 = np.convolve(self.mainWeights[0], x, 'valid')
        c0 += np.convolve(self.mainWeights[1], x2, 'valid') + self.mainBias
        c1 = x[self.windowsize // 2:self.windowsize // 2 + self.regSize] - c0
        y0 = np.convolve(self.regWeights0, c0, 'valid')[0]
        y1 = np.convolve(self.regWeights1, c1, 'valid')[0]
        #get gradients
        linGrad, quaGrad, biasGrad = self.grangerGrad(x, x2, c0, y0, c1, y1)
        lg, qg, bg, loss = self.energyGrad(x[:self.windowsize], x2[:self.windowsize], c0[0], c1[0])
        linGrad += lg * self.energyMult
        quaGrad += qg * self.energyMult
        biasGrad += bg * self.energyMult
        regGrad0, regGrad1, regBias0, regBias1, regLoss0, regLoss1 = self.regressionGrad(c0, y0, c1, y1)
        #update grads
        self.mainWeights[0] -= self.alphaMain * linGrad
        self.mainWeights[1] -= self.alphaMain * quaGrad
        self.mainBias       -= self.alphaMain * biasGrad
        self.regWeights0    -= self.alphaReg * regGrad0
        self.regBias0       -= self.alphaReg * regBias0
        self.regWeights1    -= self.alphaReg * regGrad1
        self.regBias1       -= self.alphaReg * regBias1
        #save losses
        return regLoss0, regLoss1, loss

    #x being an np.array of length windowsize, x2 being its square
    def energyGrad(self, x, x2, c0, c1):
        #update energies
        self.inputEnergy = self.energyAlpha * self.inputEnergy + x2[-1]
        self.chanelEneries[0] = self.energyAlpha * self.chanelEneries[0] + c0 * c0
        self.chanelEneries[1] = self.energyAlpha * self.chanelEneries[1] + c1 * c1
        #update gradients
            #linear
        self.energyGrads0[0] = self.energyAlpha * self.energyGrads0[0] + c0 * x
        self.energyGrads0[1] = self.energyAlpha * self.energyGrads0[1] + c1 * x
            #quadratic
        self.energyGrads1[0] = self.energyAlpha * self.energyGrads1[0] + c0 * x2
        self.energyGrads1[1] = self.energyAlpha * self.energyGrads1[1] + c1 * x2
            #bias
        self.energyGradBias[0] = self.energyAlpha * self.energyGradBias[0] + c0
        self.energyGradBias[1] = self.energyAlpha * self.energyGradBias[1] + c1
        #return current gradient
            #linear
        gradLin = (2 * self.chanelEneries[0] - self.inputEnergy) * self.energyGrads0[0]
        gradLin -= (2 * self.chanelEneries[1] - self.inputEnergy) * self.energyGrads0[1]
        gradLin *= 1 - self.energyAlpha
            #quadratic
        gradQua = (2 * self.chanelEneries[0] - self.inputEnergy) * self.energyGrads1[0]
        gradQua -= (2 * self.chanelEneries[1] - self.inputEnergy) * self.energyGrads1[1]
        gradQua *= 1 - self.energyAlpha
            #bias
        gradBias = (2 * self.chanelEneries[0] - self.inputEnergy) * self.energyGradBias[0]
        gradBias -= (2 * self.chanelEneries[1] - self.inputEnergy) * self.energyGradBias[1]
        gradBias *= 1 - self.energyAlpha
            #total loss
        loss = (self.chanelEneries[0] - self.inputEnergy / 2) ** 2 + (self.chanelEneries[1] - self.inputEnergy / 2) ** 2
        return gradLin, gradQua, gradBias, loss

    #c0 and c1 are both np.arrays of length regSize
    #y0 is the prediction of c0, y1 is the prediction of c1
    def regressionGrad(self, c0, y0, c1, y1):
        targetIdx = self.regSize // 2
        grad0 = (y0 - c1[targetIdx]) * c0
        grad1 = (y1 - c0[targetIdx]) * c1
        biasGrad0 = y0 - c1[targetIdx]
        biasGrad1 = y1 - c0[targetIdx]
        return grad0[::-1], grad1[::-1], biasGrad0, biasGrad1, biasGrad0 ** 2, biasGrad1 ** 2

    #x is an np.array with size windowsize + regSize - 1, x2 is its square
    def grangerGrad(self, x, x2, c0, y0, c1, y1):
        xSub = x[self.regSize // 2:-self.regSize // 2 + 1]
        x2Sub = x2[self.regSize // 2:-self.regSize // 2 + 1]
            #linear
        gradLin = (y0 - c0[self.regSize // 2]) * (
                    np.convolve(x, self.regWeights0, 'valid') + xSub)
        gradLin -= (y1 - c1[self.regSize // 2]) * (
                    np.convolve(x, self.regWeights1, 'valid') + xSub)
            #quadratic
        gradQua = (y0 - c0[self.regSize // 2]) * (
                np.convolve(x2, self.regWeights0, 'valid') + x2Sub)
        gradQua -= (y1 - c1[self.regSize // 2]) * (
                np.convolve(x2, self.regWeights1, 'valid') + x2Sub)
            #bias
        gradBias = (y0 - c0[self.regSize // 2]) * (sum(self.regWeights0) + 1)
        gradBias -= (y1 - c1[self.regSize // 2]) * (sum(self.regWeights1) + 1)
        return gradLin, gradQua, gradBias