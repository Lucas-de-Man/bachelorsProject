from scipy.io.wavfile import read, write
import numpy as np

for p in range(8):
    for v in range(8):
        for p_vs_v in range(1, 11):
            piano = read("wavs/piano" + str(p) + ".wav")
            violin = read("wavs/violin" + str(v) + ".wav")
            piano_audio = piano[1][:131072, 0]
            violin_audio = violin[1][:131072, 0]
            combP = piano_audio + violin_audio / 10 * p_vs_v
            combP = combP.astype(np.int16)
            if p_vs_v != 10:
                combV = piano_audio / 10 * p_vs_v + violin_audio
                combV = combV.astype(np.int16)
                write("wav_comb/comb" + str(10*p_vs_v) + "-100_" + str(p) + "-" + str(v) + ".wav", rate=44100, data=combV)
            write("wav_comb/comb100-" + str(10*p_vs_v) + "_" + str(p) + "-" + str(v) + ".wav", rate=44100, data=combP)