from midiutil.MidiFile import MIDIFile
import numpy as np

track = 0  # the only track
channel = 0


mf = MIDIFile(1)  # only 1 track
mf.addTrackName(track, 0, "Sample Track")
mf.addTempo(track, 0, 120)

maxtime = 100
for _ in range(2):
    time = 0
    while time < maxtime:
        if np.random.random() < 0.3:
            time += np.random.random() * 1.5
        else:
            duration = np.random.random() * 1.5
            duration = min(duration, maxtime - time)
            mf.addNote(track, channel, 60 + np.random.randint(-16, 16), time, duration, 100)
            time += duration
with open("midis/piano.mid", 'wb') as outf:
    mf.writeFile(outf)

mf = MIDIFile(1)  # only 1 track
mf.addTrackName(track, 0, "Sample Track")
mf.addProgramChange(track, channel, 0, 41)
mf.addTempo(track, 0, 120)

maxtime = 100
for _ in range(2):
    time = 0
    while time < maxtime:
        if np.random.random() < 0.3:
            time += np.random.random() * 1.5
        else:
            duration = np.random.random() * 1.5
            duration = min(duration, maxtime - time)
            mf.addNote(track, channel, 60 + np.random.randint(-16, 16), time, duration, 100)
            time += duration
with open("midis/violin.mid", 'wb') as outf:
    mf.writeFile(outf)