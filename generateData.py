import numpy as np
import matplotlib.pyplot as plt
import wave

#I want a minimum of 1/4th of a wave to be inside the sliding window
#and the sampling rate should be >= 2 * fmax
#7 different tones with a logarithmic scale
#so fmax = 2 * fmin, so sample rate >= 2 * fmin, making SR prime with respect to the data should help
#lowest note at 65.4 Hz, so max note at 130.8 Hz
#SR >= 261.6 Hz



"""inp = np.array(range(4073))
inp = np.cos(inp * np.pi / 4073 * 2)
plt.plot(inp)
plt.show()


wave = np.fft.fft(inp)
#wave = np.abs(wave)
plt.plot(np.abs(wave))
plt.show()
#plt.plot(np.angle(wave))
#plt.show()

print(np.abs(wave)[15:25])
print(np.angle(wave)[15:25])
"""

#making the notes

notes = np.zeros((7, 4096)) #2 * 16 * 128
fingerprint = [1, 1, 0.5, 0.25, 0.125, 0.0625, 0.03125]
nF = 1
for n in range(7):
    fac = (nF * 1.0905077326653 - nF) * 128
    for fp in range(len(fingerprint)):
        notes[n][int(fp * fac / len(fingerprint) + nF * 128)] = fingerprint[fp] * 2048
    nF *= 1.1040895136738

piano = np.empty((7, 4096))
for i in range(len(notes)):
    inp = np.fft.ifft(notes[i])
    inp = np.imag(inp)
    piano[i] = inp
    """
    inp -= min(inp)
    inp *= 2147483647. / max(inp)
    piano[i] = inp

piano = piano.astype(int)
piano = np.append(piano, piano, axis=1)"""


notes = np.zeros((7, 4096))  # 2 * 16 * 128
fingerprint = [0.5, 0.25, 0.125, 0, 0.125, 0.25, 1]
nF = 1.1
for n in range(7):
    fac = (nF * 1.0905077326653 - nF) * 128
    for fp in range(len(fingerprint)):
        notes[n][int(fp * fac / len(fingerprint) + nF * 128)] = fingerprint[fp] * 2048
    nF *= 1.1040895136738


violin = np.empty((7, 4096))
for i in range(len(notes)):
    inp = np.fft.ifft(notes[i])
    inp = np.imag(inp)
    violin[i] = inp
    """
    inp -= min(inp)
    inp *= 2147483647. / max(inp)
    violin[i] = inp

violin = violin.astype(int)
violin = np.append(violin, violin, axis=1)"""

#making music

#only going up, starting over when we hit the higest
pianoMelody = [0, 1, 2, 3, 4, 5, 6] #7
#going down but slower, skipping the last 0 to make an offset
violinMelody = [6, 6, 6, 5, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0] #20

pianoSong = np.empty(len(pianoMelody) * len(violinMelody) * piano.shape[1])
violinSong = np.empty(len(pianoMelody) * len(violinMelody) * violin.shape[1])

print(pianoSong.shape, violinSong.shape)

for i in range(len(pianoMelody) * len(violinMelody)):
    for j in range(piano.shape[1]):
        pianoSong[i * piano.shape[1] + j] = piano[pianoMelody[i % len(pianoMelody)]][j]
        violinSong[i * piano.shape[1] + j] = violin[violinMelody[i % len(violinMelody)]][j]

with open('music/music.npy', 'wb') as f:
    np.save(f, pianoSong)
    np.save(f, violinSong)
    #np.save(f, len(violin[0]) // 2)
    np.save(f, piano.shape[1])

'''
with open('music/c++Music.txt', 'w') as f:
    f.write(str(len(pianoSong)) + ' ')
    for i in range(len(pianoSong)):
        f.write(str(pianoSong[i]) + ' ')
    for i in range(len(violinSong)):
        f.write(str(violinSong[i]) + ' ')
    f.write(str(piano.shape[1]) + ' ')
'''

sumSong = pianoSong + violinSong
pianoSong -= min(pianoSong)
violinSong -= min(violinSong)
sumSong -= min(sumSong)
pianoSong *= 2147483647. / max(pianoSong)
violinSong *= 2147483647. / max(violinSong)
sumSong *= 2147483647. / max(sumSong)
pianoSong = pianoSong.astype(int)
violinSong = violinSong.astype(int)
sumSong = sumSong.astype(int)

with wave.open("violin.wav", mode='wb') as f:
    f.setnchannels(1)
    f.setsampwidth(4)
    #4096, 128 = 65.4 Hz, so 128/65.4=1.957 sec. 4096/1.957=2093 frames/sec
    f.setframerate(8192)
    f.writeframes(bytes(violinSong))

with wave.open("piano.wav", mode='wb') as f:
    f.setnchannels(1)
    f.setsampwidth(4)
    #4096, 128 = 65.4 Hz, so 128/65.4=1.957 sec. 4096/1.957=2093 frames/sec
    f.setframerate(8192)
    f.writeframes(bytes(pianoSong))

with wave.open("sum.wav", mode='wb') as f:
    f.setnchannels(1)
    f.setsampwidth(4)
    #4096, 128 = 65.4 Hz, so 128/65.4=1.957 sec. 4096/1.957=2093 frames/sec
    f.setframerate(8192)
    f.writeframes(bytes(sumSong))
