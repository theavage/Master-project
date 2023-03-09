
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
from torch.profiler import profile, record_function, ProfilerActivity
torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description= 'VERDICT training')

parser.add_argument('--acqscheme', '-trs',type=str, default = "/Users/theavage/Documents/Master/Data/GS55 - long acquisition/GS55_long_protocol2.scheme", help='Path to acquisition scheme')
parser.add_argument('--data_path','-X',type=str,default="/Users/theavage/Documents/Master/Data/GS55 - long acquisition/P55_norm.nii",help="Path to training data")
parser.add_argument('--batch_size', type=int, default = 45, help='Batch size')
parser.add_argument('--patience', '-p', type=int,default=20, help='Patience')
parser.add_argument('--epochs', '-e', type=int,default=20, help='Number of epochs')
parser.add_argument('--learning_rate', '-lr', type=float,default=0.0001, help='Learning rate')
parser.add_argument('--save_path', '-sp', type=str,default='./network/models/GS55_optimal.pt', help='models/long.pt')
parser.add_argument('--loss_path', '-lp', type=str,default='./network/models/loss_GS55_optimal.pt', help='models/long.pt')



def train_model():

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    b_values, gradient_strength, gradient_directions, delta, Delta = (get_scheme_values(args.acqscheme))
    net = Net(b_values,gradient_strength,gradient_directions,delta,Delta).to(device)
    X_train = load_data(args.data_path)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr = args.learning_rate, weight_decay=0)  

    num_batches = len(X_train) // args.batch_size
    trainloader = utils.DataLoader(X_train,
                                    batch_size = args.batch_size, 
                                    shuffle = True,
                                    num_workers = 2,
                                    drop_last = True)
    best = 1e16  
    num_bad_epochs = 0
    losses = []
    train_acc = []
    patience = args.patience
    for epoch in range(args.epochs): 
        print("-----------------------------------------------------------------")
        print("Epoch: {}; Bad epochs: {}".format(epoch, num_bad_epochs))
        net.train()
        running_loss = 0.
        for i, X_batch in enumerate(tqdm(trainloader), 0):
            optimizer.zero_grad()
            X_batch = X_batch.to(device)
            X_pred, radii,f_sphere, f_ball, f_stick = net(X_batch)
            X_pred.to(device)
            loss = criterion(X_pred.type(torch.FloatTensor), X_batch.type(torch.FloatTensor))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print("Loss: {}".format(running_loss))
        losses.append(running_loss)
        
        
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

    torch.save(final_model, args.save_path)
    #torch.save(losses, args.loss_path)


"""
args = parser.parse_args()
X_train = load_data(args.data_path)
trainloader = utils.DataLoader(X_train,
                                batch_size = args.batch_size, 
                                shuffle = True,
                                num_workers = 2,
                                drop_last = True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()
b_values, gradient_strength, gradient_directions, delta, Delta = (get_scheme_values(args.acqscheme))
net = Net(b_values,gradient_strength,gradient_directions,delta,Delta).to(device)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        for step, batch_data in enumerate(trainloader):
            if step >= (1 + 1 + 3) * 2:
                break
            net(batch_data.to(device))
        prof.step()  # Need to call this at the end of each step to notify profiler of steps' boundary.

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
"""