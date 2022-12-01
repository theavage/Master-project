import argparse
import numpy as np
import torch
import torch.nn as nn
from utils import squash, sphere_compartment, stick_compartment
from dmipy.signal_models.sphere_models import S4SphereGaussianPhaseApproximation


parser = argparse.ArgumentParser(description= 'VERDICT model')
parser.add_argument('--learningrate', '-lr', type=float, help='Learning rate')
parser.add_argument('--batchsize', '-bs', type=int, help='Batch size')
parser.add_argument('--patience', '-p', type=int, help='Patience')
parser.add_argument('--dropout', '-d', default=0, type=float, help='Dropout (0-1)')
parser.add_argument('--num_params', '-np', type=int,default=7, help='yes for s0, no for without')

args = parser.parse_args()

class Net(nn.Module):

    def __init__(self, b_values_no0,gradient_strength,gradient_directions,delta,Delta):
        super(Net, self).__init__()
        self.b_values_no0 = b_values_no0
        self.fc_layers = nn.ModuleList()
        self.gradient_strength = gradient_strength
        self.gradient_directions = gradient_directions
        self.delta = delta
        self.Delta = Delta

        for i in range(3): 
            self.fc_layers.extend([nn.Linear(len(b_values_no0), len(b_values_no0)), nn.ELU()])
        self.encoder = nn.Sequential(*self.fc_layers, nn.Linear(len(b_values_no0), 7))
        if args.dropout != 0:
            self.dropout = nn.Dropout(args.dropout)

    def forward(self, X):
        if args.dropout != 0:
            X = self.dropout(X)
        X = X.to(torch.float32)
        params = torch.abs(self.encoder(X))
        radii = squash(params[:,0],0.02,30)# Limits as set for sim data
        theta = params[:,1].unsqueeze(1)
        phi = params[:,2].unsqueeze(1)

        lambda_par = squash(params[:,3],3e-09,10e-9)
        f_sphere = squash(params[:,4],0,1)
        f_ball = squash(params[:,5],0,1)
        f_stick = squash(params[:,6],0,1)

        f_ball = 1 - (f_sphere+f_stick)
        f_stick = 1 - (f_sphere+f_ball)
        f_sphere = 1 - (f_stick + f_ball)
        lambda_iso = torch.full((radii.size()),2e-9,requires_grad=True)
        '''
        lambda_iso = torch.full((radii.size()),2e-9,requires_grad=True)
        lambda_par = torch.full((radii.size()),3e-9,requires_grad=True)
        f_ball = torch.full((radii.size()), 0.3,requires_grad=True)
        f_sphere = torch.full((radii.size()),4.,requires_grad=True)
        f_stick = torch.full((radii.size()),0.3,requires_grad=True)
        theta = torch.full((radii.size()),2.,requires_grad=True)
        phi = torch.full((radii.size()),2.,requires_grad=True)
        '''
        ball = torch.exp(-self.b_values_no0*lambda_iso)
        stick = stick_compartment(self.b_values_no0,lambda_par,self.gradient_directions,theta,phi)
        sphere = sphere_compartment(self.gradient_strength, self.delta, self.Delta, radii)

        X =  f_stick*stick + f_sphere*sphere +f_ball*ball
        return X, radii, theta, phi, lambda_par, f_sphere, f_ball, f_stick
