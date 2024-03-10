from midiutil.MidiFile import MIDIFile
import random

def piramid_notes(file, nrNotes, pitch, pitchstep, duration=1, wait=1 / 2, track=0, channel=0):
    pitch -= (nrNotes * pitchstep + pitchstep) // 2
    for n in range(nrNotes):
        file.addNote(track, channel, pitch + pitchstep * (n+1), n * wait, duration, 100)

def rev_piramid_notes(file, nrNotes, pitch, pitchstep, duration=1, wait=1 / 2, track=0, channel=0):
    pitch += (nrNotes * pitchstep + pitchstep) // 2
    for n in range(nrNotes):
        file.addNote(track, channel, pitch - pitchstep * (n+1), n * wait, duration, 100)

track = 0  # the only track
channel = 0
time = 0  # start at the beginning


for i in range(-4, 4):
    # create your MIDI object
    mf = MIDIFile(1)  # only 1 track
    mf.addTrackName(track, time, "Sample Track")
    mf.addTempo(track, time, 120)
    piramid_notes(mf, 8, 60 + 4*i, 4)
    with open("midis/piano" + str(i+4) + ".mid", 'wb') as outf:
        mf.writeFile(outf)

for i in range(-4, 4):
    # create your MIDI object
    mf = MIDIFile(1)  # only 1 track
    mf.addTrackName(track, time, "Sample Track")
    mf.addTempo(track, time, 120)
    rev_piramid_notes(mf, 8, 60 + 4*i, 4)
    with open("midis/violin" + str(i+4) + ".mid", 'wb') as outf:
        mf.writeFile(outf)