import numpy as np
import argparse
from model import Net
import torch.utils.data as utils
from utils import load_data, get_bvalues
import torch

parser = argparse.ArgumentParser(description= 'VERDICT training')

parser.add_argument('--data_path','-X',type=str,default="data/simulated_1024x160.npy",help="Path to training data")
parser.add_argument('--batch_size', type=int, default = 12, help='Batch size')
parser.add_argument('--acqscheme', '-trs',type=str, default = "/Users/theavage/Documents/Master/Data/GS55 - long acquisition/GS55_long_protocol2.scheme", help='Path to acquisition scheme')

args = parser.parse_args()

def predict(model, data):

    model.eval()  # testing mode
    res = np.empty((12,160))
    for X_batch in data:
        y_pred = model(X_batch)
        res = np.concatenate((res, y_pred.detach().numpy()), axis=0)
    return res

def make_predictions():
        
    b_values = torch.FloatTensor(get_bvalues(args.acqscheme))

    model = Net(b_values)

    X_train = load_data(args.data_path)
    testloader = utils.DataLoader(X_train,
                                        batch_size = args.batch_size, 
                                        shuffle = True,
                                        num_workers = 2,
                                        drop_last = True)

    X_test_loader= next(iter(testloader))

    pred = predict(model, testloader) 

    return pred
