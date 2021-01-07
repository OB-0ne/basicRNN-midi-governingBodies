import torch
import matplotlib.pyplot as plt


def save_checkpoint(net, optimizer, epoch_no, loss, checkpoint_name="", store_eNum = True):

    path = "saved_net/"
    if checkpoint_name == "":
        path = path + "checkpoint_e" + str(epoch_no+1) + ".pt"
    else:
        path = path + checkpoint_name
        if store_eNum:
            path = path + "_e" + str(epoch_no+1)
        path += ".pt"

    checkpoint = {}
    checkpoint = {'epoch': epoch_no,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'meta': update_nn_metaData()
                  }

    torch.save(checkpoint, path)

    print(f"----- Saved the network as '{path}' -----")

def load_checkpoint(net, optimizer, checkpoint_name, net_evalMode = False):

    path = "saved_net/"
    path = path + checkpoint_name + ".pt"

    checkpoint = torch.load(path)

    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_no = checkpoint['epoch']
    loss = checkpoint['loss']
    nn_meta = checkpoint['meta']

    apply_nn_metaData(nn_meta)

    if net_evalMode:
        net.eval()
    else:
        net.train()

    print(f"----- Loaded the network from '{path}' -----")

    return net, optimizer, epoch_no, loss

def load_checkpoint_generator(checkpoint_name):

    path = "saved_net/"
    path = path + checkpoint_name + ".pt"

    checkpoint = torch.load(path)

    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_no = checkpoint['epoch']
    loss = checkpoint['loss']
    nn_meta = checkpoint['meta']

    apply_nn_metaData(nn_meta)

    net.eval()

    print(f"----- Loaded the network from '{path}' -----")

    return net, optimizer, epoch_no, loss


def apply_nn_metaData(nn_meta):

    input_size = nn_meta['input_size']
    midi_features = nn_meta['midi_features']
    feature_size = nn_meta['feature_size']
    hidden_size = nn_meta['hidden_size']
    output_size = nn_meta['output_size']
    num_layers = nn_meta['num_layers']
    dropout_per = nn_meta['dropout_per']
    learning_rate = nn_meta['learning_rate']
    batch_size = nn_meta['batch_size']

def update_nn_metaData():

    nn_meta = {}

    nn_meta['input_size'] = input_size
    nn_meta['midi_features'] = midi_features
    nn_meta['feature_size'] = feature_size
    nn_meta['hidden_size'] = hidden_size
    nn_meta['output_size'] = output_size
    nn_meta['num_layers'] = num_layers
    nn_meta['dropout_per'] = dropout_per
    nn_meta['learning_rate'] = learning_rate
    nn_meta['batch_size'] = batch_size

    return nn_meta

def generate_sample_song(song_length_seconds, song_name = "test_output.wav", showSignal = False, saveMIDI = False, saveNumpy = True, seed = 35):

    # variables for the song output
    total_iterations = song_length_seconds

    # get the features for the seed
    input_seed = torch.FloatTensor(all_feature_matrix[seed:seed+input_size]).to(device)

    # make a zero variable to input the song into
    song = np.zeros((song_length_seconds,4))
    song[0:input_size] = input_seed.cpu().detach().numpy()

    # set the network to evaluation mode
    net.eval()

    # loop through the needed iterations
    for i in range(total_iterations-input_size):

        input_seed = torch.FloatTensor(song[i:input_size + i]).to(device)

        # get the output from the network
        nn_output = net(input_seed)

        # add the current output to the song
        song[input_size + i] = nn_output.cpu().detach().numpy()

    if saveNumpy:
        np.save(song_name,np.array(song))

    if saveMIDI:
        DM = DataManager()
        DM.np2MIDI(song, song_name,AutoTimed=True)
        