import mido
import numpy as np

max_midi_time = 1000.0

def standardizeData(midiData):
    for msg in midiData:
        msg[1] = float(msg[1])/127.0
        msg[2] = float(msg[2])/127.0
        msg[3] = float(msg[3])/max_midi_time
    return midiData


mid = mido.MidiFile('song.mid')

i = 0
total_time = 0.0
mid_out = []

for i,track in enumerate(mid.tracks):
        
    for msg in track:

        if msg.type == "control_change":
            #skip for now I guess
            continue
        elif msg.type == "note_on":
            mid_out.append([1,msg.note,msg.velocity,msg.time])
        elif msg.type == "note_off":
            mid_out.append([0,msg.note,msg.velocity,msg.time])

    mid_out = np.array(standardizeData(mid_out))
    np.save("out_" + str(i),np.array(mid_out))
