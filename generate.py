# set loading values as variables
sample_MIDI_size = 80
total_samples = 15
total_seeds = all_feature_matrix.shape[0]//input_size

# load a checkpoint
checkpoint_name = "heavyRain_e500"

# make the network and put it on GPU
net = MidiRNN().float().to(device)
optimizer = torch.optim.Adamax(net.parameters(), lr=learning_rate)

watch = StopWatch()

net, _, _, _ = load_checkpoint(net,optimizer,checkpoint_name)

for i in range(total_samples):
    generate_sample_song(sample_MIDI_size, f'outputs_run/heavyRain_timed_{(i+1):02d}',saveMIDI=True, saveNumpy=False, seed=random.randint(0, total_seeds), AutoTimed=True)
    print(f'{watch.give()} heavyRain_timed_{(i+1):02d} generated')