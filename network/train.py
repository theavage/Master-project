
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from tqdm import tqdm
from model import Net


parser = argparse.ArgumentParser(description= 'VERDICT training')

parser.add_argument('--acqscheme', '-trs',type=str, default = "/Users/theavage/Documents/Master/Data/GS55 - long acquisition/GS55_long_protocol2.scheme", help='Path to acquisition scheme')
parser.add_argument('--Xdata','-X',type=str,default="data/simulated_data.npz",help="Path to training data")
parser.add_argument('--batch_size', type=int, default = 12, help='Batch size')
parser.add_argument('--patience', '-p', type=int,default=10, help='Patience')
parser.add_argument('--epochs', '-e', type=int,default=1000, help='Number of epochs')
parser.add_argument('--learning_rate', '-lr', type=float,default=0.0001, help='Learning rate')

args = parser.parse_args()

def train_model():

    net = Net("/Users/theavage/Documents/Master/Data/GS55 - long acquisition/GS55_long_protocol2.scheme")
    X_train = np.load(args.Xdata)['arr_0']

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr = args.learning_rate)  
    batch_size = args.batch_size

    num_batches = len(X_train) // batch_size
    trainloader = utils.DataLoader(torch.from_numpy(X_train.astype(np.float32)),
                                    batch_size = batch_size, 
                                    shuffle = True,
                                    num_workers = 2,
                                    drop_last = True)
    best = 1e16
    num_bad_epochs = 0
    patience = args.patience

    for epoch in range(args.epochs): 
        print("-----------------------------------------------------------------")
        print("Epoch: {}; Bad epochs: {}".format(epoch, num_bad_epochs))
        net.train()
        running_loss = 0.

        for i, X_batch in enumerate(tqdm(trainloader), 0):
            optimizer.zero_grad()
            X_pred, radii, mu1, mu2, lambda_par, f_sphere, f_ball, f_stick = net(X_batch)
            loss = criterion(X_pred.type(torch.FloatTensor), X_batch.type(torch.FloatTensor))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        
        print("Loss: {}".format(running_loss))

        if running_loss < best:
            print("############### Saving good model ###############################")
            final_model = net.state_dict()
            best = running_loss
            num_bad_epochs = 0
        else:
            num_bad_epochs = num_bad_epochs + 1
            if num_bad_epochs == patience:
                print("Done, best loss: {}".format(best))
                final_model = net.state_dict()
                break
            print("Done")
    net.load_state_dict(final_model)

train_model()