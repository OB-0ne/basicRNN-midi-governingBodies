import nn_util
import numpy as np
from DataManager import DataManager as dmMIDI

def test1():
    song = np.load('midiTest.npy')

    for x in song:
        x[3] = x[3]*1000/580

    np.save('midi_test_new',song)

def test2():
    song = np.load('midi_test_new.npy')
    print(song)

test2()