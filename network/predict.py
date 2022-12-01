import numpy as np
import argparse
from model import Net
import torch.utils.data as utils
from utils import load_data, get_scheme_values
import torch

parser = argparse.ArgumentParser(description= 'VERDICT training')

parser.add_argument('--data_path','-X',type=str,default="data/simulated_1024x120.npy",help="Path to training data")
parser.add_argument('--batch_size', type=int, default = 64, help='Batch size')
parser.add_argument('--acqscheme', '-trs',type=str, default = "/Users/theavage/Documents/Master/Data/GS55 - long acquisition/GS55_long_protocol2.scheme", help='Path to acquisition scheme')
parser.add_argument('--model_path', '-mpt',type=str, default = "/Users/theavage/Documents/Master/Master-project/network/models/test_model_50_120.pt", help='Path to model')

args = parser.parse_args()

def predict(model, data):

    model.eval()  # testing mode
    X_res = np.empty((args.batch_size,120))
    radii_res = np.empty((args.batch_size,1))
    f_sphere_res = np.empty((args.batch_size,1))
    f_stick_res = np.empty((args.batch_size,1))
    f_ball_res = np.empty((args.batch_size,1))
    for X_batch in data:
        X,radii, mu1, mu2, lambda_par, f_sphere, f_ball, f_stick = model(X_batch)
        X_res = np.concatenate((X_res, X.detach().numpy()), axis=0)
        radii_res = np.concatenate((radii_res, radii.detach().numpy()), axis=0)
        f_sphere_res = np.concatenate((f_sphere_res, f_sphere.detach().numpy()), axis=0)
        f_stick_res = np.concatenate((f_stick_res, f_stick.detach().numpy()), axis=0)
        f_ball_res = np.concatenate((f_ball_res, f_ball.detach().numpy()), axis=0)
    return X_res, radii_res,f_sphere_res,f_ball_res, f_stick_res

def make_predictions():
        
    b_values, gradient_strength, gradient_directions,delta, Delta = (get_scheme_values(args.acqscheme))

    model = Net(b_values,gradient_strength,gradient_directions,delta,Delta)
    model.load_state_dict(torch.load(args.model_path))

    X_train = load_data(args.data_path)
    testloader = utils.DataLoader(X_train,
                                        batch_size = args.batch_size, 
                                        shuffle = False,
                                        num_workers = 2,
                                        drop_last = True)


    X,radii,f_sphere, f_ball,f_stick = predict(model, testloader) 

    return X,radii,f_sphere, f_ball,f_stick