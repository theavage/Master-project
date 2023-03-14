import numpy as np
import argparse
from model import Net
import torch.utils.data as utils
from utils import load_data, get_scheme_values
import torch

parser = argparse.ArgumentParser(description= 'VERDICT training')

parser.add_argument('--acqscheme', '-trs',type=str, default = "./data/GS55_long_protocol2.scheme", help='Path to acquisition scheme')
parser.add_argument('--data_path','-X',type=str,default="./data/GS55_norm.nii.gz",help="Path to training data")
parser.add_argument('--batch_size', type=int, default = 50, help='Batch size')
parser.add_argument('--model_path', '-mpt',type=str, default = "./network/models/model_160.pt", help='Path to model')
parser.add_argument('--mask_path','-m',type=str,default="./data/GS55_all_mask.nii",help="Path to training data")


args = parser.parse_args()

def predict(model, data, mask):

    model.eval()  # testing mode
    X_res = np.empty((args.batch_size,data.dataset.data.shape[1]))
    radii_res = np.empty((args.batch_size,1))
    f_sphere_res = np.empty((args.batch_size,1))
    f_stick_res = np.empty((args.batch_size,1))
    f_ball_res = np.empty((args.batch_size,1))

    for X_batch, mask_batch in zip(data, mask):
        if mask_batch.sum().sum()==0:
            X = np.zeros([args.batch_size,data.dataset.data.shape[1]])
            radii, f_sphere, f_ball, f_stick = np.zeros([4,args.batch_size,1])
            X_res = np.concatenate((X_res, X), axis=0)
            radii_res = np.concatenate((radii_res, radii), axis=0)
            f_sphere_res = np.concatenate((f_sphere_res, f_sphere), axis=0)
            f_stick_res = np.concatenate((f_stick_res, f_stick), axis=0)
            f_ball_res = np.concatenate((f_ball_res, f_ball), axis=0)
        else:
            X,radii, f_sphere, f_ball, f_stick = model(X_batch)
            X_res = np.concatenate((X_res, X.cpu().detach().numpy()), axis=0)
            radii_res = np.concatenate((radii_res, radii.cpu().detach().numpy()), axis=0)
            f_sphere_res = np.concatenate((f_sphere_res, f_sphere.cpu().detach().numpy()), axis=0)
            f_stick_res = np.concatenate((f_stick_res, f_stick.cpu().detach().numpy()), axis=0)
            f_ball_res = np.concatenate((f_ball_res, f_ball.cpu().detach().numpy()), axis=0)
    return X_res[args.batch_size:], radii_res[args.batch_size:,:],f_sphere_res[args.batch_size:],f_ball_res[args.batch_size:], f_stick_res[args.batch_size:]

def make_predictions():
        
    b_values, gradient_strength, gradient_directions,delta, Delta = (get_scheme_values(args.acqscheme))

    model = Net(b_values,gradient_strength,gradient_directions,delta,Delta)
    model.load_state_dict(torch.load(args.model_path))

    X_train, mask = load_data(args.data_path, args.mask_path)
    testloader = utils.DataLoader(X_train,
                                        batch_size = args.batch_size, 
                                        shuffle = False,
                                        num_workers = 2,
                                        drop_last = False)
    
    maskloader = utils.DataLoader(mask,
                                        batch_size = args.batch_size, 
                                        shuffle = False,
                                        num_workers = 2,
                                        drop_last = False)


    X,radii,f_sphere, f_ball,f_stick = predict(model, testloader,maskloader) 

    return X,radii,f_sphere, f_ball,f_stick