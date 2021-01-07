from DataManager import DataManager as midiDM

# MAIN CODE HERE

DM = DataManager()

# example to convert numpy to MIDI
# DM.npFile2MIDI('midi_e7800.npy','MIDI out/track1',AutoTimed=True,AutoTime=180)
# DM.npFile2MIDI('midi_e5150.npy','MIDI out/track2',AutoTimed=True,AutoTime=140)

# example to convert MIDI to numpy
DM.MIDIFile2np('data/beatles1.mid','data/out')

# [OB][NOTE]:
# Autotime can also modify the rhythm, try time signatures