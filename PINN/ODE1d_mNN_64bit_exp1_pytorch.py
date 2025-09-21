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