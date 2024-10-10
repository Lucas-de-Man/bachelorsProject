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

pianoSong = np.empty(50 * piano.shape[1])
violinSong = np.empty(50 * violin.shape[1])

print(pianoSong.shape, violinSong.shape)

i = 0
while i < len(pianoSong):
    k = np.random.randint(0, 7)
    l = np.random.randint(1, 5) #1, 2, 3, 4
    if i + len(piano[k]) * l // 4 >= len(pianoSong):
        l = (len(pianoSong) - i) / len(piano[k]) * 4
    for j in range(int(len(piano[k]) * l // 4)):
        pianoSong[i + j] = piano[k][j]
    i += piano.shape[1] * l // 4

i = 0
while i < len(violinSong):
    k = np.random.randint(0, 7)
    l = np.random.randint(1, 5) #1, 2, 3, 4
    if i + len(violin[k]) * l // 4 >= len(violinSong):
        l = (len(violinSong) - i) / len(violin[k]) * 4
    for j in range(int(len(violin[k]) * l // 4)):
        violinSong[i + j] = violin[k][j]
    i += violin.shape[1] * l // 4


with open('music/music.npy', 'wb') as f:
    np.save(f, pianoSong)
    np.save(f, violinSong)
    np.save(f, len(violin[0]) // 2)


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
