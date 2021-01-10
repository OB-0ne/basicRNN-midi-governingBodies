import numpy as np
from MIDI_nn import MidiRNN
from utils import StopWatch
import torch
import nn_util
import random

# get the data
#read the fetures if not in memeory
all_feature_matrix = np.load("data/heavyRain.npy")
print("MIDI Reading Completed!")
print("Shape of the features: ", all_feature_matrix.shape)

# load config files
config = nn_util.load_config(generate_config=True)

# setting the device to run the code to GPU is avaialble
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# set loading values as variables
sample_MIDI_size = config['sample_MIDI_size']
total_samples = config['total_samples']
total_seeds = all_feature_matrix.shape[0]//config['input_size']

# load a checkpoint
checkpoint_name = config['checkpoint_name']

# make the network and put it on GPU
net = MidiRNN(config).float().to(device)
optimizer = torch.optim.Adamax(net.parameters(), lr=config['learning_rate'])

watch = StopWatch()

net, _, _, _, config = nn_util.load_checkpoint(net,optimizer,checkpoint_name,net_evalMode = True)

for i in range(total_samples):
    nn_util.generate_sample_song(config, net, all_feature_matrix, device, sample_MIDI_size, f'outputs/heavyRain_{(i+1):02d}',saveMIDI=True, saveNumpy=False, seed=random.randint(0, total_seeds), AutoTimed=False)
    print(f'{watch.give()} heavyRain_{(i+1):02d} generated')