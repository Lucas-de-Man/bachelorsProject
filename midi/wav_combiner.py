from scipy.io.wavfile import read, write
import numpy as np

#making the music mean 0
pianoArr = []
violinArr = []
for p in range(8):
    piano = read("wavs/piano" + str(p) + ".wav")[1][:131072, 0]
    piano = piano - np.mean(piano)
    pianoArr.append(piano)
for v in range(8):
    violin = read("wavs/piano" + str(v) + ".wav")[1][:131072, 0]
    violin = violin - np.mean(violin)
    violinArr.append(violin)

for p, piano in enumerate(pianoArr):
    for v, violin in enumerate(violinArr):
        for p_vs_v in range(1, 11):
            combP = piano + violin / 10 * p_vs_v
            combP = combP.astype(np.int16)
            if p_vs_v != 10:
                combV = piano / 10 * p_vs_v + violin
                combV = combV.astype(np.int16)
                write("wav_comb/comb" + str(10*p_vs_v) + "-100_" + str(p) + "-" + str(v) + ".wav", rate=44100, data=combV)
            write("wav_comb/comb100-" + str(10*p_vs_v) + "_" + str(p) + "-" + str(v) + ".wav", rate=44100, data=combP)