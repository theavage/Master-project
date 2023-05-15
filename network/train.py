"""

Script training neural network model, 
with input functions from utils.py and model.py

"""


import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
import torch.nn.functional as F
from tqdm import tqdm
from model import Net
from utils import load_data, get_scheme_values

#Release all unoccupied cached memory
torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description= 'VERDICT training')

parser.add_argument('--acqscheme', '-trs',type=str, default = "./data/GS55_long_protocol2.scheme", help='Path to acquisition scheme')
parser.add_argument('--data_path','-X',type=str,default="./data/simulated_9180_160.npy",help="Path to training data")
parser.add_argument('--mask_path','-m',type=str,default=None,help="Path to training data")
parser.add_argument('--batch_size', type=int, default = 45, help='Batch size')
parser.add_argument('--patience', '-p', type=int,default=20, help='Patience')
parser.add_argument('--epochs', '-e', type=int,default=50, help='Number of epochs')
parser.add_argument('--learning_rate', '-lr', type=float,default=0.0001, help='Learning rate')
parser.add_argument('--save_path', '-sp', type=str,default='./network/models/model_160.pt', help='models/long.pt')
parser.add_argument('--loss_path', '-lp', type=str,default='./network/models/loss_160.pt', help='models/long.pt')


def train_model():
    """
    
    Does the entire training process, including data loading, passing the
    data through the network for X epochs and saving the trained model.
    
    """

    args = parser.parse_args()

    # Makes sure the model runs on GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the acquisition scheme values
    b_values, gradient_strength, gradient_directions, delta, Delta = (get_scheme_values(args.acqscheme))

    # Loading the model and sends it to device
    net = Net(b_values,gradient_strength,gradient_directions,delta,Delta).to(device)

    # Dataloading
    X_train = load_data(args.data_path, args.mask_path)
    trainloader = utils.DataLoader(X_train,
                                    batch_size = args.batch_size, 
                                    shuffle = True,
                                    num_workers = 2,
                                    drop_last = False)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr = args.learning_rate, weight_decay=0) 

    # Coutnters
    best = 1e16  
    num_bad_epochs = 0
    patience = args.patience

    # List for saving the training loss
    losses = []

    # Performing the forward and backward passes for X amount of epochs
    for epoch in range(args.epochs): 

        print("-----------------------------------------------------------------")
        print("Epoch: {}; Bad epochs: {}".format(epoch, num_bad_epochs))
        net.train() # Training mode
        running_loss = 0. # Initializing

        # Looping through all data in batches
        for i, X_batch in enumerate(tqdm(trainloader), 0):

            # zero all gradients
            optimizer.zero_grad()

            # Sending batch to device
            X_batch = X_batch.to(device)

            # Predictions from model
            X_pred, radii,f_sphere, f_ball, f_stick = net(X_batch) 

            # Sending predictions ot device
            X_pred.to(device)

            # Computing loss 
            loss = criterion(X_pred.type(torch.FloatTensor), X_batch.type(torch.FloatTensor)) 

            # Backward pass
            loss.backward() 

            # Updating weights
            optimizer.step() 

            # Updating loss
            running_loss += loss.item() 

        print("Loss: {}".format(running_loss))
        losses.append(running_loss)
        
        # Decide if traning should be stopped, or if it should go to next epoch
        if running_loss < best:
            print("############### Saving good model ###############################")
            final_model = net.state_dict()
            best = running_loss
            num_bad_epochs = 0
        else:
            num_bad_epochs = num_bad_epochs + 1
            if num_bad_epochs == patience:
                print("Done, best loss: {}".format(best))
                break
            print("Done")
    
    # Saving the trained model
    torch.save(final_model, args.save_path)

