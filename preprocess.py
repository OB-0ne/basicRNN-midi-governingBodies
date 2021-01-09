from DataManager import DataManager as midiDM
import numpy as np

# example to convert numpy to MIDI
# DM.npFile2MIDI('midi_e7800.npy','MIDI out/track1',AutoTimed=True,AutoTime=180)
# DM.npFile2MIDI('midi_e5150.npy','MIDI out/track2',AutoTimed=True,AutoTime=140)

# example to convert MIDI to numpy
song1 = midiDM('data/midi_heavyRain/01.mid')
song2 = midiDM('data/midi_heavyRain/02.mid')
song3 = midiDM('data/midi_heavyRain/03.mid')

song1.save_np('data/midi_heavyRain_processed/01')
song2.save_np('data/midi_heavyRain_processed/02')
song3.save_np('data/midi_heavyRain_processed/03')

songs = [song1, song2, song3]
x = []
for song in songs:
    x.extend(list(song.mid_npy))
    x.extend([[0,0,0,0]]*10)

x = np.array(x)
np.save('data/heavyRain',np.array(x))


# DM.MIDIFile2np('data/midi_heavyRain/02.mid','data/midi_heavyRain_processed/02.mid')
# DM.MIDIFile2np('data/midi_heavyRain/03.mid','data/midi_heavyRain_processed/03.mid')

# [OB][NOTE]:
# Autotime can also modify the rhythm, try time signatures