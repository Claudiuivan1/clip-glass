import torch
from torch import Tensor
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
        
        
class L_ReLU( Module ):
    
    # define negative slope
    a = 0

    # initialize class variables
    def __init__( self, a = 0.1 ):
        super( L_ReLU, self ).__init__()
        self.a = a

    # forward application method
    def forward( self, input: Tensor ) -> Tensor:
        # apply LReLU
        x_pos = input
        x_neg = torch.mul( x_pos, self.a )
        y = torch.where( x_pos > 0, x_pos, x_neg )
        
        return y
        
        
class BiR_ReLU( Module ):
    
    # define default bounds for random distribution
    lower_bound = 0
    upper_bound = 0

    # initialize class variables
    def __init__( self, a_lower_bound = 0.1, a_upper_bound = 0.2, b_lower_bound = 0.8, b_upper_bound = 1.0 ):
        super( BiR_ReLU, self ).__init__()
        self.a_lower_bound = a_lower_bound
        self.a_upper_bound = a_upper_bound
        self.b_lower_bound = b_lower_bound
        self.b_upper_bound = b_upper_bound

    # forward application method
    def forward( self, input: Tensor ) -> Tensor:
        # obtain a and b from uniform random distribution
        a = np.random.uniform( self.a_lower_bound, self.a_upper_bound )
        b = np.random.uniform( self.b_lower_bound, self.b_upper_bound )
        # apply BiRReLU
        x_pos = input
        x_neg = torch.mul( x_pos, a )
        x_pos = torch.mul( x_pos, b )
        y = torch.where( x_pos > 0, x_pos, x_neg )
        
        return y
        
     
class BiL_ReLU( Module ):
    
    # define negative slope
    a = 0

    # initialize class variables
    def __init__( self, a = 0.1, b = 0.9 ):
        super( BiL_ReLU, self ).__init__()
        self.a = a
        self.b = b

    # forward application method
    def forward( self, input: Tensor ) -> Tensor:
        x_pos = input
        x_neg = torch.mul( x_pos, self.a )
        x_pos = torch.mul( x_pos, self.b )
        y = torch.where( x_pos > 0, x_pos, x_neg )
        
        return y