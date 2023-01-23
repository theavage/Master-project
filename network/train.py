
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
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description= 'VERDICT training')

parser.add_argument('--acqscheme', '-trs',type=str, default = "/Users/theavage/Documents/Master/Master-project/data/3466.scheme", help='Path to acquisition scheme')
parser.add_argument('--data_path','-X',type=str,default="/Users/theavage/Documents/Master/Master-project/data/simulated_3466.npy",help="Path to training data")
parser.add_argument('--batch_size', type=int, default = 32, help='Batch size')
parser.add_argument('--patience', '-p', type=int,default=20, help='Patience')
parser.add_argument('--epochs', '-e', type=int,default=10, help='Number of epochs')
parser.add_argument('--learning_rate', '-lr', type=float,default=0.0001, help='Learning rate')
parser.add_argument('--save_path', '-sp', type=str,default='/Users/theavage/Documents/Master/Master-project/network/models/model_3466_50_fast.pt', help='models/long.pt')

args = parser.parse_args()

def train_model():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    b_values, gradient_strength, gradient_directions, delta, Delta = (get_scheme_values(args.acqscheme))
    net = Net(b_values,gradient_strength,gradient_directions,delta,Delta).to(device)
    X_train = load_data(args.data_path)
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr = args.learning_rate)  

    num_batches = len(X_train) // args.batch_size
    trainloader = utils.DataLoader(X_train,
                                    batch_size = args.batch_size, 
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
            X_batch = X_batch.to(device)
            X_pred, radii, theta, phi, lambda_par, f_sphere, f_ball, f_stick = net(X_batch)
            X_pred.to(device)
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
                break
            print("Done")


    #plt.plot(radii_values,'r')
    #plt.savefig('radii.png')

    #plt.plot(np.array(f_ball_values),'r')
    #plt.savefig('f_ball.png')

    torch.save(final_model, args.save_path)

