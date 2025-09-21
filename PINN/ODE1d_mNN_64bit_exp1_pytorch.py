import torch
import torch.nn as nn 
import numpy as np 

import time
import functools
from scipy.io import savemat
from pathlib import Path


rootdir = Path(__file__).parent

torch.set_default_dtype(torch.float64)

def init_MLP(manual_seed,layer_widths):
    base_gen = torch.Generator()
    base_gen.manual_seed(manual_seed)
    params = []
    new_seeds = torch.randint(0,2**16-1,(len(layer_widths)-1,),generator=base_gen).tolist()
    keys =[(torch.Generator().manual_seed(s),torch.Generator().manual_seed(2*s)) for s in new_seeds]
    for in_dim, out_dim,(weight_key, bias_key) in zip(layer_widths[:-1], layer_widths[1:],keys):
        
        xavier_std = np.sqrt(2/(in_dim + out_dim))


        weights = torch.randn((out_dim, in_dim), generator=weight_key)
        
        # Reject and resample weights outside the range [-2, 2]
        while torch.any(weights > 2) or torch.any(weights < -2):
            mask = (weights > 2) | (weights < -2)
            new_weights = torch.randn(torch.sum(mask).item(), generator=weight_key)
            weights[mask] = new_weights

            
        weights = nn.Parameter(weights * xavier_std)
        
        # Truncated normal initialization for biases
        biases = torch.randn((out_dim,), generator=bias_key)

        while torch.any(biases > 2) or torch.any(biases < -2):
            mask = (biases > 2) | (biases < -2)
            new_biases = torch.randn(torch.sum(mask).item(), generator=bias_key)
            biases[mask] = new_biases

        biases = nn.Parameter(biases * xavier_std)

        params.append([weights, biases])
    return params

def neural_net(params, z, limit, scl, act_s=0):
    '''
    :param params: weights and biases
    :param x: input data [matrix with shape [N, m]]; m is number of inputs)
    :param limit: characteristic scale for normalizeation [matrx with shape [2, m]]
    :param sgn:  1 for even function and -1 for odd function
    :return: neural network output [matrix with shape [N, n]]; n is number of outputs)
    '''
    lb = limit[0]  # lower bound for each input
    ub = limit[1]  # upper bound for each input

    actv = torch.tanh()

    H = 2.0 * (z - lb) / (ub - lb) - 1.0  # input normalization

    first, *hidden, last = params

    H = actv(torch.matmul(H, first[0].T) * scl + first[1])
    
    # Calculate the middle layers output
    for layer in hidden:
        H = torch.tanh(torch.matmul(H, layer[0].T) + layer[1])
    
    # No activation function for last layer
    var = torch.matmul(H, last[0].T) + last[1]
    return var


def sech(z):
    return 1/torch.cosh(z)

def sol_init_MLP(manual_seed, n_hl, n_unit):
    '''
    :param n_hl: number of hidden layers [int]
    :param n_unit: number of units in each layer [int]
    '''
    layers = [1] + n_hl * [n_unit] + [1]
    # generate the random key for each network
    # generate weights and biases for
    params_u = init_MLP(manual_seed, layers)
    return dict(net_u=params_u)


# wrapper to create solution function with given domain size
def sol_pred_create(limit, scl, act_s=0):
    '''
    :param limit: domain size of the input
    :return: function of the solution (a callable)
    '''
    def f_u(params, z):
        # generate the NN
        u = neural_net(params['net_u'], z, limit, scl, act_s)
        return u
    return f_u



def mNN_pred_create(f_u, limit, scl, epsil, act_s=0):
    '''
    :param f_u: sum of previous stage network
    :param limit: domain size of the input
    :return: function of the solution (a callable)
    '''
    def f_comb(params, z):
        # generate the NN
        u_now = neural_net(params['net_u'], z, limit, scl, act_s)
        u = f_u(z) + epsil * u_now
        return u
    return f_comb


def ms_error(diff):
    return torch.mean(diff**2, dim=0)

