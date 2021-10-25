import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '..', 'deep_prior'))

from networks.ae import  AutoencoderSelu
from utils import *
import torch
import scipy.io
import torch.nn as nn


default_model_path = os.path.join(os.path.join(dir_path, '..', 'deep_prior/trained_models/map_network.model'))

# Out of range model
revision_model_path =  os.path.join(os.path.join(dir_path, '..', 'deep_prior/trained_models/revision1_benign_map.model'))

# full_map_model = torch.load(model_path)

def map_complete(X, W, model_path='default'):
    if not model_path == 'default':
        full_map_model  = torch.load(model_path)
    else:
        full_map_model = torch.load(default_model_path)

    X = torch.from_numpy(X).type(torch.float32)
    W = torch.from_numpy(W).type(torch.float32)
    K = X.shape[2]

    W = W.unsqueeze(dim=0)
    W = W.unsqueeze(dim=0)
    Wx = W.repeat(K,1,1,1)
    Wx[Wx<0.5] = 0
    Wx[Wx>=0.5] = 1

    X = X.permute(2,0,1)
    X = X.unsqueeze(dim=1)
    X_sampled = X*Wx
    test_map = torch.cat((Wx,X_sampled), dim=1)

    full_map = full_map_model(test_map)
    full_map = full_map.squeeze()
    full_map = full_map.permute(1,2,0)
    full_map = full_map.detach().numpy()

    return full_map.copy()

if __name__ == '__main__':
    # X = np.random.rand(51,51,64)
    # W = np.ones((51,51))
    # z = np.random.rand(51,51,5)
    # C = np.random.rand(64,5)
    # R = 5
    # a = run_descent(W,X,z,C,R)
    # ROOT = '/home/sagar/Projects/radio_map_deep_prior/psd_recovery/data'
    # BASE = '/home/sagar/Projects/radio_map_deep_prior/deep_prior'

    # X = scipy.io.loadmat(os.path.join(ROOT,'T.mat'))['T']
    # W = scipy.io.loadmat(os.path.join(ROOT,'Om.mat'))['Om']
    # full_map = model(X,W)
    # print(cost_func(X,full_map))
    pass