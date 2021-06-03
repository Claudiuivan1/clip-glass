'''
import torch
from torch import Tensor
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter

'''

import torch
from torch.nn import Module
import numpy as np

class R_ReLU( Module ):
    
    # define default bounds for random distribution
    lower_bound = 0
    upper_bound = 0

    # initialize class variables
    def __init__( self, lower_bound = 0.1, upper_bound = 0.2 ):
        super( R_ReLU, self ).__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    # forward application method
    def forward( self, input: Tensor ) -> Tensor:
        # obtain a from uniform random distribution
        a = np.random.uniform( self.lower_bound, self.upper_bound )
        # apply RReLU
        x_pos = input
        x_neg = torch.mul( x_pos, a )
        y = torch.where( x_pos > 0, x_pos, x_neg )
        
        return y