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

#note length
noteLength = 2048

notes = np.zeros((4, 4096)) #2 * 16 * 128
fingerprint = [1]#, 1, 1, 0.5, 0.25, 0.125]
nF = 8
expStep = np.power(2, 1 / notes.shape[0])
for n in range(notes.shape[0]):
    fac = 1
    for fp in range(len(fingerprint)):
        notes[n][int(fac * nF)] = fingerprint[fp] * 2048
        #we multiply fac by a factor s.t. the current note stops before the start of the next note
        fac *= np.power(expStep, 1 / len(fingerprint))
    nF *= expStep

#take only the beginning of the note (end early)
piano = np.empty((notes.shape[0], noteLength))
for i in range(notes.shape[0]):
    inp = np.fft.ifft(notes[i])
    inp = np.imag(inp)
    piano[i] = inp[:piano.shape[1]]
    """
    inp -= min(inp)
    inp *= 2147483647. / max(inp)
    piano[i] = inp

piano = piano.astype(int)
piano = np.append(piano, piano, axis=1)"""


notes = np.zeros((4, 4096))  # 2 * 16 * 128
fingerprint = [1]#0.125, 0, 0.125, 0.25, 1]
#1.127... is a random number to offset violin and piano on the freq spectrum
nF = 1.1345324 * 16#1.127896458325 * 32
expStep = np.power(2, 1 / notes.shape[0])
for n in range(notes.shape[0]):
    fac = 1
    for fp in range(len(fingerprint)):
        notes[n][int(fac * nF)] = fingerprint[fp] * 2048
        #we multiply fac by a factor s.t. the current note stops before the start of the next note
        fac *= np.power(expStep, 1 / len(fingerprint))
    nF *= expStep

#take only the beginning of the note (end early)
violin = np.empty((notes.shape[0], noteLength))
for i in range(notes.shape[0]):
    inp = np.fft.ifft(notes[i])
    inp = np.imag(inp)
    violin[i] = inp[:violin.shape[1]]
    """
    inp -= min(inp)
    inp *= 2147483647. / max(inp)
    violin[i] = inp

violin = violin.astype(int)
violin = np.append(violin, violin, axis=1)"""

#making music

#only going up, starting over when we hit the higest
pianoMelody = [0, 1, 2, 3] #4
#going down but slower, skipping the last 0 to make an offset
violinMelody = [3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0] #11

'''
pianoSong = np.zeros(len(pianoMelody) * len(violinMelody) * piano.shape[1])
violinSong = np.zeros(len(pianoMelody) * len(violinMelody) * violin.shape[1])

print(pianoSong.shape, violinSong.shape)


for i in range(len(pianoMelody) * len(violinMelody)):
    for j in range(piano.shape[1]):
        pianoSong[i * piano.shape[1] + j] = piano[pianoMelody[i % len(pianoMelody)]][j]
        violinSong[i * violin.shape[1] + j] = violin[violinMelody[i % len(violinMelody)]][j]
'''

#infinitly smooth 0 <= t <= 1
def smoothStep(t):
    if t < 0.001 or t > 0.999:
        return t
    et = np.exp(-1 / t)
    return et / (et + np.exp(1/(t - 1)))

#returns intensety as a funciton of time 0 <= t <= 1
def intensity(t, maxAt=0.1):
    total = (t <= maxAt) * smoothStep(t / maxAt)
    total += (t > maxAt) * (t < 1 - maxAt)      #1 when not stepping
    return total + (t >= 1 - maxAt) * smoothStep((1 - t) / maxAt)

monoProp = 0.7
pianoSong = np.zeros(len(pianoMelody) * len(violinMelody) * int(piano.shape[1] * monoProp))
violinSong = np.zeros(len(pianoMelody) * len(violinMelody) * int(violin.shape[1] * monoProp))

'''
data = [intensity(i / piano.shape[1], (1 - monoProp)/2) for i in range(piano.shape[1])]
plt.plot(data)
plt.show()
'''

print(pianoSong.shape, violinSong.shape)

for i in range(len(pianoMelody) * len(violinMelody)):
    for j in range(piano.shape[1]):
        pianoSong[(i * int(piano.shape[1] * monoProp) + j) % len(pianoSong)] += piano[pianoMelody[i % len(pianoMelody)]][j] * intensity(j / piano.shape[1], (1 - monoProp)/2)
        violinSong[(i * int(violin.shape[1] * monoProp) + j) % len(violinSong)] += violin[violinMelody[i % len(violinMelody)]][j] * intensity(j / piano.shape[1], (1 - monoProp)/2)



def meanEnergy(data):
    return np.dot(data, data) / len(data)

#make sure the mean energies of both signals are the same
violinSong *= np.sqrt(meanEnergy(pianoSong) / meanEnergy(violinSong))

with open('music/music.npy', 'wb') as f:
    np.save(f, pianoSong)
    np.save(f, violinSong)
    #max relevant shift
    np.save(f, piano.shape[1] * 7)

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
