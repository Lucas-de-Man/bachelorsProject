import numpy as np

class Grengar:
    def __init__(self, windowsize=256, regSize=64, alphaMain=0.005, magMult=1, alphaReg=0.01, orthAlpha=0.99, verbose=False):
        self.windowsize = windowsize
        self.regSize = regSize
        self.alphaMain = alphaMain
        self.magMult = magMult
        self.alphaReg = alphaReg
        self.verbose = verbose
        self.orthAlpha = orthAlpha

            #weigts
        #main weights have linear and quadratic components and a bias
        self.mainWeights = [np.random.normal(0, 1 / windowsize, windowsize) for _ in range(2)]
        self.mainBias = np.random.normal(0, 1 / windowsize)
        self.regWeights0 = np.random.normal(0, 1 / regSize, regSize)
        self.regBias0 = np.random.normal(0, 1 / regSize)
        self.regWeights1 = np.random.normal(0, 15 / regSize, regSize)
        self.regBias1 = np.random.normal(0, 1 / regSize)

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

    def regGrad(self, chanel0, chanel1):
        if len(chanel0) < self.regSize:
            print("input needs to be bigger than or equal to the regression size")
            return
                                                                  #predict the middle value of the other chanel
        diff0 = np.convolve(self.regWeights0, chanel0, 'valid') + self.regBias0 - chanel1[self.regSize // 2:-self.regSize + self.regSize // 2 + 1]
        diff1 = np.convolve(self.regWeights1, chanel1, 'valid') + self.regBias1 - chanel0[self.regSize // 2:-self.regSize + self.regSize // 2 + 1]
        #reverse to counteract future convolving
        diff0 = diff0[::-1]
        diff1 = diff1[::-1]
        grad0 = np.convolve(diff0, chanel0, 'valid')
        grad1 = np.convolve(diff1, chanel1, 'valid')
        return grad0, grad1, diff0, diff1

    def orthGrad(self, input, inputSquare, chanel0, chanel1):
        sqSum = chanel0 ** 2 + chanel1 ** 2
        diff = chanel1 - chanel0

        linGrad = np.zeros(len(self.mainWeights[0]))
        sqGrad  = np.zeros(len(self.mainWeights[1]))
        biasGrad = 0
        lastBias = 0
        lastScaleLin = 0
        lastScaleSq = 0
        lastScaleBi = 0
        for t in range(len(diff)):
            lastCL = np.zeros(len(self.mainWeights[0]))
            lastCS = np.zeros(len(self.mainWeights[0]))
            lastScaleLin = self.orthAlpha * lastScaleLin + inputSquare[t] * sqSum[t]
            lastScaleSq = self.orthAlpha * lastScaleSq + inputSquare[t] ** 2 * sqSum[t]
            lastScaleBi = self.orthAlpha * lastScaleBi + sqSum[t]
            lastBias = self.orthAlpha * lastBias + diff[t]
            for j in range(len(self.mainWeights[0])):
                lastCL[j] = self.orthAlpha * lastCL[j] + diff[t] * input[t + j]
                lastCS[j] = self.orthAlpha * lastCS[j] + diff[t] * inputSquare[t + j]
            linGrad += lastCL * lastScaleLin
            sqGrad += lastCS * lastScaleSq
            biasGrad += lastBias * lastScaleBi
        return 2 * (1 - self.orthAlpha) ** 2 * linGrad, 2 * (1 - self.orthAlpha) ** 2 * sqGrad, 2 * (1 - self.orthAlpha) ** 2 * biasGrad


    def step(self, input):
        chanel0, chanel1 = self.forward(input)
        regGrad0, regGrad1, diff0, diff1 = self.regGrad(chanel0, chanel1)
        inpSq = input ** 2
        linGrad, sqGrad, biasGrad = self.orthGrad(input, inpSq, chanel0, chanel1)
        #orthoganality
        self.mainWeights[0] += linGrad / len(input) * self.alphaMain * self.magMult
        self.mainWeights[1] += sqGrad / len(input) * self.alphaMain * self.magMult
        self.mainBias += biasGrad / len(input) * self.alphaMain * self.magMult
        #linear terms
        mainGrad0 = np.convolve(self.regWeights0, input, 'valid')
        mainGrad1 = np.convolve(self.regWeights1, input, 'valid')
        alignedInput = input[self.regSize // 2:-self.regSize + self.regSize // 2 + 1]
        mainGrad0 += alignedInput
        mainGrad1 += alignedInput
        self.mainWeights[0] += np.convolve(diff0, mainGrad0, 'valid')[::-1] / len(input) * self.alphaMain
        self.mainWeights[0] -= np.convolve(diff1, mainGrad1, 'valid')[::-1] / len(input) * self.alphaMain
        #quadratic terms
        qmainGrad0 = np.convolve(self.regWeights0, inpSq, 'valid')
        qmainGrad1 = np.convolve(self.regWeights1, inpSq, 'valid')
        qalignedInput = inpSq[self.regSize // 2:-self.regSize + self.regSize // 2 + 1]
        qmainGrad0 += qalignedInput
        qmainGrad1 += qalignedInput
        self.mainWeights[1] += np.convolve(diff0, qmainGrad0, 'valid')[::-1] / len(input) * self.alphaMain
        self.mainWeights[1] -= np.convolve(diff1, qmainGrad1, 'valid')[::-1] / len(input) * self.alphaMain
        #main bias
        self.mainBias += sum(diff0) * (sum(self.regWeights0) + self.regBias0 + 1) / len(input) * self.alphaMain
        self.mainBias -= sum(diff1) * (sum(self.regWeights1) + self.regBias1 + 1) / len(input) * self.alphaMain
        # regressions
        self.regWeights0 -= regGrad0[::-1] / len(input) * self.alphaReg
        self.regWeights1 -= regGrad1[::-1] / len(input) * self.alphaReg
        self.regBias0 -= sum(diff0) / len(input) * self.alphaReg
        self.regBias1 -= sum(diff1) / len(input) * self.alphaReg
        if self.verbose:
            #print loss if verbose (MSE-ish)
            sqDiff = inpSq[self.windowsize // 2:-self.windowsize + self.windowsize // 2 + 1] - chanel0 ** 2 - chanel1 ** 2
            print("orth: ", sum(sqDiff) ** 2 / len(input))
            print("rest: ", -sum(diff0**2 + diff1**2) / len(input))
            print("orthGrad: ", sum(abs(linGrad) + abs(sqGrad)) / len(linGrad) * self.magMult)
            mainSq = np.convolve(diff0, qmainGrad0, 'valid')[::-1] - np.convolve(diff1, qmainGrad1, 'valid')[::-1]
            mainLin = np.convolve(diff0, mainGrad0, 'valid')[::-1] - np.convolve(diff1, mainGrad1, 'valid')[::-1]
            print("restGrad: ", sum(abs(mainLin) + abs(mainSq)) / len(mainLin))
            print("-------------------")