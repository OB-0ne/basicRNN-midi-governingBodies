import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utils import StopWatch
import nn_util

from MIDI_nn import MidiRNN
import matplotlib.pyplot as plt

import numpy as np
import math


def validate_network(showError = False):

    # making a list of all the batch number which belong to the testing groups
    test_list = [x for x in range(total_train,total_train+total_test)]
    loss_by_batch = []

    # setting the network t evaluation
    net.eval()

    # iterate through the testing bacthes
    for i in range(total_train,total_train + total_test - input_size - 1):

        # set loss to to zero after each batch iteration
        loss = 0

        # get the needed input and actual output values 
        input_matrix = torch.FloatTensor(all_feature_matrix[i:i + input_size]).to(device)
        val_output = torch.FloatTensor(np.array(all_feature_matrix[i+input_size+1])).to(device)

        # get the network output
        nn_output = net(input_matrix)

        # check the network output and add the loss
        loss += loss_function(nn_output, val_output)

        # add the loss to a list which contains loss for all batches
        loss_by_batch.append(loss)

    # plot the graph of the batch loss as a line graph
    if showError:
        plt.plot(loss_by_batch)
        plt.ylabel('Loss by batch')
        plt.show()

# load config files
config = nn_util.load_config()

# Setting up variables for the neural networks
input_size = config['input_size']
batch_size = config['batch_size']

# setting the device to run the code to GPU is avaialble
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# set the number of epoch and traininng perecntage of the dataset
epochs = config['epochs']
training_per = 0.9
start_epoch = config['start_epoch']

# load a checkpoint
load = True
save_checkpoint_name = 'heavyRain_pc'
checkpoint_name = "heavyRain_pc_e1"

# get the data
#read the fetures if not in memeory
all_feature_matrix = np.load("data/heavyRain.npy")
print("MIDI Reading Completed!")
print("Shape of the features: ", all_feature_matrix.shape)

# this calculates the total number of chucks to be used for training and testing
total_train = int(all_feature_matrix.shape[0] * config['training_per'])
total_test = all_feature_matrix.shape[0] - total_train

#get total batch sizes and math around it
total_batches = math.ceil(total_train/batch_size)

# make the network and put it on GPU
net = MidiRNN(config).float().to(device)

# define an optimizer and loss function
# this can be changed as per the model
optimizer = torch.optim.Adamax(net.parameters(), lr=config['learning_rate'])
loss_function = nn.MSELoss()

# making a stopwatch to count time
watch = StopWatch()

if load:
    net, optimizer, start_epoch, loss, config = nn_util.load_checkpoint(net,optimizer,checkpoint_name)
    start_epoch = start_epoch + 1

# loop for all the epochs
for epoch in range(start_epoch,epochs):

    # reset the epoch loss
    epoch_loss = 0

    for batch in range(total_batches):
        # reset the hidden layers and remove all gradients after each batch iteration, which also considers back propogation
        net.reset_hidden()
        net.zero_grad()
        loss = 0 

        batch_start = max((batch * batch_size) - input_size,0)
        batch_end = max(((batch+1) * batch_size),total_train) - input_size

        # run for all the chunks
        for i in range(batch_start, batch_end):
                
            # make the input and validation output tensors
            input_matrix = torch.FloatTensor(all_feature_matrix[i:i + input_size]).to(device)
            val_output = torch.FloatTensor(np.array(all_feature_matrix[i+input_size+1])).to(device)

            # get the network output
            nn_output = net(input_matrix)

            # calculate the loss from the nnetwork output and valid output
            loss += loss_function(nn_output, val_output.view(1,1,-1))
            epoch_loss += loss

        # back propogate through the network with the accumulated error and optimizer
        loss.backward()
        optimizer.step()

        if (batch+1) % 5 == 0:
            # a print to now the end of an epoch and its loss
            print(f'{watch.give()} Epoch {epoch + 1} Batch {batch + 1} Batch Loss: {round(float(loss),6)}')

    # a print to now the end of an epoch and its loss
    print(f'---- {watch.give()} Epoch {epoch + 1} completed! Total Loss: {round(float(epoch_loss),6)}')

    # after a few epochs check with the testing of the network and also generate a song sample
    if (epoch+1) % config['test_network'] == 0:
        validate_network(showError=False)
        nn_util.generate_sample_song(config, net, all_feature_matrix, device, 50, f'outputs/heavyRain_e{epoch+1}',saveMIDI = True, saveNumpy=False)
        nn_util.save_checkpoint(net,config,optimizer,epoch,loss,checkpoint_name=save_checkpoint_name)
        net.train()