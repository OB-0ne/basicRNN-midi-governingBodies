import mido
import numpy as np
from pathlib import Path

class DataManager:

    mid = None
    file_location = None
    mid_npy = None

    max_midi_time = 580.0
    AutoTime = 480
    
    def __init__(self, sFileName):
        if Path(sFileName).exists() and sFileName!='':
            self.file_location = sFileName
                        
            self.read_MIDI()
            self.midi2np()
        
    def read_MIDI(self):
        self.mid = mido.MidiFile(self.file_location)

    def print_MIDI(self):
        for i, track in enumerate(self.mid.tracks):
            print('---- Track ',str(i+1),":", track.name, '----')
            for msg in track:
                print(msg)

    def midi2np(self):
        
        def standardizeData(midiData):

            max_midi_time = 580.0

            for msg in midiData:
                msg[1] = float(msg[1])/127.0
                msg[2] = float(msg[2])/127.0
                msg[3] = float(msg[3])/self.max_midi_time
                
            return midiData

        i = 0
        mid_out = []

        for i,track in enumerate(self.mid.tracks):
            for msg in track:
                if msg.type == "control_change":
                    #skip for now I guess
                    continue
                elif msg.type == "note_on":
                    mid_out.append([1,msg.note,msg.velocity,msg.time])
                elif msg.type == "note_off":
                    mid_out.append([0,msg.note,msg.velocity,msg.time])

        self.mid_npy = np.array(standardizeData(mid_out))
        
        
    def save_np(self, sFilename):
        np.save(sFilename,np.array(self.mid_npy))
    
    
    ### OLD CODE ###
    def npFile2MIDI(self, in_filename, out_filename, num = 4, den = 4, clocks = 36, noted32 = 8, AutoTimed = False, AutoTime=120):
            
        max_midi_time = 1000.0

        data = np.load(in_filename)
        mid = mido.MidiFile()
        track = mido.MidiTrack()

        mid.tracks.append(track)
        
        num = 4
        den = 4
        clocks = 36
        noted32 = 8

        track.append(mido.MetaMessage('time_signature', numerator=num, denominator=den, clocks_per_click=clocks, notated_32nd_notes_per_beat=noted32, time=0))
        test=[]

        def convert127(msg):
            note = int(msg*127)
            note = min(note,127)
            note = max(note,0)

            return note

        def convertTime(msg):
            time_out = abs(int(msg*max_midi_time))
            time_out = int(min(time_out, max_midi_time))
            return time_out

        for msg in data:

            if int(msg[0]+0.5) == 1:
                control = 'note_on'
            else:
                control = 'note_off'
            
            if AutoTimed:
                track.append(mido.Message(control, note=convert127(msg[1]), velocity=convert127(msg[2]), time=AutoTime))
                
            else:
                track.append(mido.Message(control, note=convert127(msg[1]), velocity=convert127(msg[2]), time=convertTime(msg[3])))

        
        if not out_filename[-4] == '.mid':
            out_filename += '.mid'

        mid.save(out_filename)

    def np2MIDI(self, np_track, out_filename, num = 4, den = 4, clocks = 36, noted32 = 8, AutoTimed = False, AutoTime=120):
            
        max_midi_time = 1000.0

        data = np_track
        mid = mido.MidiFile()
        track = mido.MidiTrack()

        mid.tracks.append(track)
        
        num = 4
        den = 4
        clocks = 36
        noted32 = 8

        track.append(mido.MetaMessage('time_signature', numerator=num, denominator=den, clocks_per_click=clocks, notated_32nd_notes_per_beat=noted32, time=0))
        test=[]

        for msg in data:

            if int(msg[0]+0.5) == 1:
                control = 'note_on'
            else:
                control = 'note_off'
            
            if AutoTimed:
                track.append(mido.Message(control, note=int(msg[1]*127), velocity=int(msg[2]*127), time=AutoTime))
                
            else:
                track.append(mido.Message(control, note=int(msg[1]*127), velocity=int(msg[2]*127), time=int(msg[3]*max_midi_time)))

        if not out_filename[-4] == '.mid':
            out_filename += '.mid'

        mid.save(out_filename)

    def MIDIFile2np(self, in_filename, out_filname):
        max_midi_time = 1000.0

        def standardizeData(midiData):
            for msg in midiData:
                msg[1] = float(msg[1])/127.0
                msg[2] = float(msg[2])/127.0
                msg[3] = float(msg[3])/max_midi_time
            return midiData


        mid = read_MIDI(in_filename)

        i = 0
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
            np.save(out_filname + str(i),np.array(mid_out))

    