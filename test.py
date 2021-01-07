import mido
import numpy as np

mid = mido.MidiFile('sample_01.mid')
# mid = mido.MidiFile('song2.mid')
i=0
for t in mid.tracks:
    for msg in t:
        print(msg)

    if i >= 20:
        break

    i = i + 1

# midi = np.load('midi_e150.npy')
# for a in midi:
#     print(a)

# for msg in mid.play():
#     print(msg)