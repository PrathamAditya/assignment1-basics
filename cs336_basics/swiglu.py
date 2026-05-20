from .linear import Linear
import torch.nn as nn
import math
import torch
import numpy as np

class SwiGLU(nn.Module):

    def _get_d_ff(self, d_model):
        target = (8 / 3) * d_model
        return math.ceil(target / 64) * 64

    def _get_SiLU(self, x):
        return x * torch.sigmoid(x)
    
    def __init__(self, d_model: int, d_ff = None):
        super().__init__()
        self.d_model = d_model

        if d_ff is None:
            self.d_ff = self._get_d_ff(self.d_model)
        else: 
            self.d_ff = d_ff
            
        self.w3 = Linear(self.d_model, self.d_ff)
        self.w1 = Linear(self.d_model, self.d_ff)
        self.w2 = Linear(self.d_ff, self.d_model)
        
    
    def forward(self, x: torch.Tensor):
        branch1 = self._get_SiLU(self.w1(x))
        branch2 = self.w3(x)
        gated = torch.mul(branch1, branch2)
        return self.w2(gated)
       
        

    
    

    



