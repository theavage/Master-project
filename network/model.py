import argparse
import numpy as np
import torch
import torch.nn as nn
from utils import squash #helper functions

parser = argparse.ArgumentParser(description= 'VERDICT model')
parser.add_argument('--learningrate', '-lr', type=float, help='Learning rate')
parser.add_argument('--batchsize', '-bs', type=int, help='Batch size')
parser.add_argument('--patience', '-p', type=int, help='Patience')
parser.add_argument('--dropout', '-d', default=1, type=float, help='Dropout (0-1)')
parser.add_argument('--s0', '-s', type=str, help='yes for s0, no for without')
parser.add_argument('--parameters', '-np', type=int,default=6, help='yes for s0, no for without')

args = parser.parse_args()

class Net(nn.Module):
    def __init__(self, ti_no0, gradient_directions_no0, b_values_no0):
        super(Net, self).__init__()
        self.ti_no0 = ti_no0
        self.gradient_directions_no0 = gradient_directions_no0
        self.b_values_no0 = b_values_no0
        self.fc_layers = nn.ModuleList()
        for i in range(3): 
            self.fc_layers.extend([nn.Linear(len(b_values_no0), len(b_values_no0)), nn.ELU()])
        self.encoder = nn.Sequential(*self.fc_layers, nn.Linear(len(b_values_no0), args.parameters))
        if args.dropout != 0:
            self.dropout = nn.Dropout(args.dropout)
    
    def forward(self, X):
        if args.dropout != 0:
            X = self.dropout(X)
        params = torch.abs(self.encoder(X))
        


        t1_ball_uns = params[:, 0]
        t1_ball = squash(t1_ball_uns, 0.010, 5.0)
        t1_stick_uns = params[:, 1]
        t1_stick = squash(t1_stick_uns, 0.010, 5.0)
        lambda_par_uns = params[:, 2]
        lambda_par = squash(lambda_par_uns, 0.1, 3.0)
        lambda_iso_uns = params[:, 3]
        lambda_iso = squash(lambda_iso_uns, 0.1, 3.0)
        Fp = params[:,6].unsqueeze(1)
        theta = params[:,4].unsqueeze(1)
        phi = params[:,5].unsqueeze(1)
        mu_cart = torch.zeros(3,X.size()[0])
        sintheta = torch.sin(theta)
        mu_cart[0,:] = torch.squeeze(sintheta * torch.cos(phi))
        mu_cart[1,:] = torch.squeeze(sintheta * torch.sin(phi))
        mu_cart[2,:] = torch.squeeze(torch.cos(theta))
        if args.s0 == 'yes':
            s0 = params[:,7].unsqueeze(1)
        else: s0 = torch.ones_like(t1_ball)
        mm_prod =  torch.einsum("ij,jk->ki",self.gradient_directions_no0, mu_cart)
        X = (Fp*(torch.abs(1 - (2*torch.exp(-self.ti_no0/t1_ball)) + torch.exp(-7.5/t1_ball))*torch.exp(-self.b_values_no0 * lambda_iso)) + (1-Fp)*(torch.abs(1 - (2*torch.exp(-self.ti_no0/t1_stick)) + torch.exp(-7.5/t1_stick))*torch.exp(-self.b_values_no0 * lambda_par * mm_prod ** 2)))*s0
        return X, t1_ball, t1_stick, lambda_par, lambda_iso, mu_cart, Fp, s0