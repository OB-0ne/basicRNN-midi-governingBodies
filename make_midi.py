import numpy as np
from nn_util import load_checkpoint, generate_sample_song
from stopWatch import StopWatch
from random import random


# get the data and oinput size
all_feature_matrix = np.load('data/test.MIDI.npy')
input_size = 20

# set loading values as variables
sample_MIDI_size = 1000
total_samples = 5
total_seeds = all_feature_matrix.shape[0]//input_size

# load a checkpoint
checkpoint_name = "checkpoint_e50"

watch = StopWatch()

net, _, _, _ = load_checkpoint(net,'fake_optimizer',checkpoint_name, runOnly=True)

for i in range(total_samples):
    generate_sample_song(net, sample_MIDI_size, f'outputs_run/sample_{(i+1):02d}',saveMIDI = True, saveNumpy=False, seed=random.randint(0, total_seeds))
    print(f'{watch.give()} sample_{(i+1):02d} generated')