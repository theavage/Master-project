import argparse
import numpy as np
import torch
import torch.nn as nn
from utils import squash, simulate_signal_dmipy


parser = argparse.ArgumentParser(description= 'VERDICT model')
parser.add_argument('--learningrate', '-lr', type=float, help='Learning rate')
parser.add_argument('--batchsize', '-bs', type=int, help='Batch size')
parser.add_argument('--patience', '-p', type=int, help='Patience')
parser.add_argument('--dropout', '-d', default=1, type=float, help='Dropout (0-1)')
parser.add_argument('--s0', '-s', type=str, help='yes for s0, no for without')
parser.add_argument('--parameters', '-np', type=int,default=7, help='yes for s0, no for without')

args = parser.parse_args()

class Net(nn.Module):
    def __init__(self, path_to_acqscheme):
        super(Net, self).__init__()

        self.fc_layers = nn.ModuleList()
        self.path_to_acqscheme = path_to_acqscheme

        for i in range(3): 
            self.fc_layers.extend([nn.Linear(160, 160), nn.ELU()]) #Hardcoding
        self.encoder = nn.Sequential(*self.fc_layers, nn.Linear(160, args.parameters))
        if args.dropout != 0:
            self.dropout = nn.Dropout(args.dropout)
    
    def forward(self, X):
        if args.dropout != 0:
            X = self.dropout(X)
        params = torch.abs(self.encoder(X))

        radii = squash(params[:,0],0.02,30).detach().numpy() # Limits as set for sim data
        mu1 = params[:,1].unsqueeze(1).detach().numpy()
        mu2 = params[:,2].unsqueeze(1).detach().numpy()
        lambda_par = params[:,3].unsqueeze(1).detach().numpy()
        f_sphere = squash(params[:,4],0,1).detach().numpy()
        f_ball = squash(params[:,5],0,1).detach().numpy()
        f_stick = squash(params[:,6],0,1).detach().numpy()

        parameter_array = np.array([radii,mu1,mu2,lambda_par,f_sphere,f_ball,f_stick])

        synthdata = simulate_signal_dmipy(self.path_to_acqscheme,parameter_array)
        synthdata = torch.tensor(synthdata,requires_grad=True)

        return synthdata, radii, mu1, mu2, lambda_par, f_sphere, f_ball, f_stick
