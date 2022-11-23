import argparse
import numpy as np
import torch
import torch.nn as nn
from utils import squash
from dmipy.signal_models.sphere_models import S4SphereGaussianPhaseApproximation


parser = argparse.ArgumentParser(description= 'VERDICT model')
parser.add_argument('--learningrate', '-lr', type=float, help='Learning rate')
parser.add_argument('--batchsize', '-bs', type=int, help='Batch size')
parser.add_argument('--patience', '-p', type=int, help='Patience')
parser.add_argument('--dropout', '-d', default=0, type=float, help='Dropout (0-1)')
parser.add_argument('--s0', '-s', type=str, help='yes for s0, no for without')
parser.add_argument('--num_params', '-np', type=int,default=7, help='yes for s0, no for without')

args = parser.parse_args()

class Net(nn.Module):

    def __init__(self, b_values_no0):
        super(Net, self).__init__()
        self.b_values_no0 = b_values_no0
        self.fc_layers = nn.ModuleList()
        for i in range(3): 
            self.fc_layers.extend([nn.Linear(len(b_values_no0), len(b_values_no0)), nn.ELU()])
        self.encoder = nn.Sequential(*self.fc_layers, nn.Linear(len(b_values_no0), args.num_params))
        if args.dropout != 0:
            self.dropout = nn.Dropout(args.dropout)
    
    def forward(self, X):
        if args.dropout != 0:
            X = self.dropout(X)
        X = X.to(torch.float32)
        params = torch.abs(self.encoder(X))

        radii = squash(params[:,0],0.02,30)# Limits as set for sim data
        mu1 = params[:,1].unsqueeze(1)
        mu2 = params[:,2].unsqueeze(1)
        lambda_par = params[:,3].unsqueeze(1)
        f_sphere = squash(params[:,4],0,1)
        f_ball = squash(params[:,5],0,1)
        f_stick = squash(params[:,6],0,1)

        #parameter_array = [radii,mu1,mu2,lambda_par,f_sphere,f_ball,f_stick]

        #synthdata = simulate_signal_dmipy(self.path_to_acqscheme,parameter_array)
        #synthdata = torch.tensor(synthdata,requires_grad=True)

        ball = f_ball*torch.exp(-self.b_values_no0*2e-9)
        stick = f_stick*torch.exp(-self.b_values_no0*lambda_par*mu1**2)

        X = ball+stick
        return X, radii, mu1, mu2, lambda_par, f_sphere, f_ball, f_stick
