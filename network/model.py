import argparse
import numpy as np
import torch
import torch.nn as nn
from utils import squash, sphere_compartment, stick_compartment, fractions_to_1
from dmipy.signal_models.sphere_models import S4SphereGaussianPhaseApproximation


parser = argparse.ArgumentParser(description= 'VERDICT model')
parser.add_argument('--dropout', '-d', default=0.5, type=float, help='Dropout (0-1)')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):

    def __init__(self, b_values_no0,gradient_strength,gradient_directions,delta,Delta):
        super(Net, self).__init__()
        self.b_values_no0 = b_values_no0.to(device)
        self.fc_layers = nn.ModuleList()
        self.gradient_strength = gradient_strength.to(device)
        self.gradient_directions = gradient_directions.to(device)
        self.delta = delta.to(device)
        self.Delta = Delta.to(device)

        for i in range(3):
            self.fc_layers.extend([nn.Linear(len(b_values_no0), len(b_values_no0)), nn.ELU()])
        self.encoder = nn.Sequential(*self.fc_layers, nn.Linear(len(b_values_no0), 4))
        if args.dropout != 0:
            self.dropout = nn.Dropout(args.dropout)

    def forward(self, X):
        if args.dropout != 0:
            X = self.dropout(X.float())
        X = X.to(torch.float32)
        params = torch.abs(self.encoder(X))
        radii = params[:,0].unsqueeze(1).to(device)
        theta = torch.full((radii.size()),1.570796326794897,requires_grad=True,device=device)
        phi = torch.full((radii.size()),0.0,requires_grad=True,device=device)

        #lambda_par = squash(params[:,3],3e-09,10e-9)
        lambda_par = torch.full((radii.size()),8e-9,requires_grad=True,device=device)
        lambda_iso = torch.full((radii.size()),2e-9,requires_grad=True,device=device)

        f_sphere,f_ball,f_stick = fractions_to_1(params[:,1],params[:,2],params[:,3])
        
        ball = torch.exp(-self.b_values_no0.to(device)*lambda_iso.to(device))
        stick = stick_compartment(self.b_values_no0,lambda_par,self.gradient_directions,theta,phi)
        sphere = sphere_compartment(self.gradient_strength, self.delta, self.Delta, radii)

        X =  f_stick*stick + f_sphere*sphere +f_ball*ball

        return X, radii, f_sphere, f_ball, f_stick
