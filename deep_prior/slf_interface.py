import sys
import time
import os
# sys.path.append('/scratch/sagar/Projects/matlab/radio_map_deep_prior/deep_prior')
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '..', 'deep_prior'))
# sys.path.append('/home/sagar/Projects/deep_spectrum_cartography/deep_prior')

from networks.ae import  AutoencoderSelu
from utils import *
import torch
import torch.nn as nn

lr = 0.01
loop_count = 10
criterion = nn.MSELoss()

default_model_path = os.path.join(os.path.join(dir_path, '..', 'deep_prior/trained_models/model1.pt'))


def nasdac_complete(S_omega, W, R, model_path='default'):
    autoencoder = AutoencoderSelu()
    autoencoder.eval()
    if not model_path == 'default':
        checkpoint  = torch.load(model_path)
    else:
        checkpoint = torch.load(default_model_path)
    autoencoder.load_state_dict(checkpoint['model_state_dict'])

    S_omega = torch.from_numpy(S_omega).type(torch.float32)
    if S_omega.dim() == 2:
        S_omega = S_omega.unsqueeze(dim=-1)
    W = torch.from_numpy(W).type(torch.float32)

    R = int(R)
    W = W.unsqueeze(dim=0)
    W = W.unsqueeze(dim=0)
    Wr = W.repeat(R,1,1,1)
    Wr[Wr<0.5] = 0
    Wr[Wr>=0.5] = 1
    S_omega = S_omega.permute(2,0,1)
    S_omega = S_omega.unsqueeze(dim=1)

    test_slf = torch.cat((Wr, S_omega), dim=1)
    slf_complete = autoencoder(test_slf)

    slf_comp = slf_complete[:,0,:,:]
    slf_comp = slf_comp.permute(1,2,0)
    slf_comp = slf_comp.detach().numpy()
    return slf_comp.copy()

def dowjons_get_initial_z(S_omega, W, R, model_path='default'):
    autoencoder = AutoencoderSelu()
    autoencoder.eval()
    if not model_path == 'default':
        checkpoint  = torch.load(model_path)
    else:
        checkpoint = torch.load(default_model_path)
    autoencoder.load_state_dict(checkpoint['model_state_dict'])

    S_omega= torch.from_numpy(S_omega).type(torch.float32)
    if S_omega.dim() == 2:
        S_omega = S_omega.unsqueeze(dim=-1)
    W = torch.from_numpy(W).type(torch.float32)

    R = int(R)
    W = W.unsqueeze(dim=0)
    W = W.unsqueeze(dim=0)
    Wr = W.repeat(R,1,1,1)
    Wr[Wr<0.5] = 0
    Wr[Wr>=0.5] = 1
    S_omega = S_omega.permute(2,0,1)
    S_omega = S_omega.unsqueeze(dim=1)

    test_slf = torch.cat((Wr, S_omega), dim=1)
    slf_complete = autoencoder(test_slf)
    Z = autoencoder.encoder(test_slf)

    slf_comp = slf_complete[:,0,:,:]
    slf_comp = slf_comp.permute(1,2,0)
    slf_comp = slf_comp.detach().numpy()
    return Z.detach().numpy().copy(), slf_comp.copy()


def optimize_s(W, X, Z, C, R, lambda_reg=0, model_path='default'):
    """
    Arguments:
        W : Mask 
        X : sampled tensor
        z : current latent vectors estimate for R emitters
        C : current psd estimate

    Returns:
        the updated latent vector estimate
    """
    autoencoder = AutoencoderSelu()
    autoencoder.eval()
    if not model_path == 'default':
        checkpoint  = torch.load(model_path)
    else:
        checkpoint = torch.load(default_model_path)
    autoencoder.load_state_dict(checkpoint['model_state_dict'])

    # Initialization need not be timed as it can be avoided by implementing everything in python
    # Data conversion to torch
    W = torch.from_numpy(W).type(torch.float32)
    X = torch.from_numpy(X).type(torch.float32)
    Z = torch.from_numpy(Z).type(torch.float32)
    C = torch.from_numpy(C).type(torch.float32)
    R = int(R)

    K = X.shape[2]

    X = X.permute(2,0,1)
    # z = z.permute(2,0,1)
    # z = z.unsqueeze(dim=1)
    print('Shape of C', C.shape)
    if C.dim() == 1:
        C = C.unsqueeze(dim=-1)
    C = C.permute(1,0)
    print(f'Shape of C {C.shape} and Z {Z.shape}')
    W = W.unsqueeze(dim=0)
    W = W.unsqueeze(dim=0)
    Wr = W.repeat(R,1,1,1)
    Wx = W.repeat(K,1,1,1)

    Wr[Wr<0.5] = 0
    Wr[Wr>=0.5] = 1

    # Z_est = torch.cat((Wr, z), dim=1)
    Z_est = Z    
    Z_est.requires_grad = True
    
    slf_complete = autoencoder.decoder(Z_est)
    X_from_slf = get_tensor(slf_complete[:,0,:,:], C)
    
    optimizer = torch.optim.Adam([Z_est], lr=0.01)

    t1 = time.time()
    for i in range(loop_count):
        optimizer.zero_grad()
        slf_complete = autoencoder.decoder(Z_est)
     
        X_from_slf = get_tensor(slf_complete[:,0,:,:], C)
        loss = cost_func(X, X_from_slf, Wx) + lambda_reg*torch.norm(Z_est)
        loss.backward()
        optimizer.step()
    t1 = time.time() - t1
    slf_comp = slf_complete[:,0,:,:]
    slf_comp = slf_comp.permute(1,2,0)
    slf_comp = slf_comp.detach().numpy()

    return Z_est.detach().numpy().copy(), slf_comp.copy(), t1


if __name__ == '__main__':
    pass
