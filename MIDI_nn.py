import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# definfng the Neural network class
class MidiRNN(nn.Module):

    # initializaing the network
    # declaring all the needed layers
    def __init__(self, config):
        super(MidiRNN,self).__init__()

        # set variables for later
        self.feature_size = config['feature_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.output_size = config['output_size']
        self.dropout_per = config['dropout_per']

        # an lstm layer for input to hidden layers
        self.rnn = nn.LSTM(self.feature_size, self.hidden_size, self.num_layers)
        # hidden to putput
        self.out = nn.Linear(self.hidden_size, self.output_size)
        # a dropdout layer between the hideen and output layer 
        self.drop = nn.Dropout(p=self.dropout_per)

        # making the hidden layer and setting it to zero
        self.hidden = ((torch.zeros(self.num_layers, 1, self.hidden_size)), (torch.zeros(self.num_layers, 1, self.hidden_size)))

    def reset_hidden(self):
        # resetting the hidden layer to zero, which can be done after backpropogation
        self.hidden = ((torch.zeros(self.num_layers, 1, self.hidden_size)), (torch.zeros(self.num_layers, 1, self.hidden_size)))

    #setting the network layers in order
    def forward(self, seq):
        # here, the view is adding anoher dimention to the sequence being passed to the network
        out, self.hidden = self.rnn(seq.view(1,1,-1))
        # out, self.hidden = self.rnn(seq.view(1,feature_size,-1))
        out = self.drop(out)
        # out = self.out(out.view(1,-1))
        out = self.out(out)

        return out