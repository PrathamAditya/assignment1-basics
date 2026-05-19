import torch
import torch.nn as nn
import math
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features:int, device: torch.device=None, dtype: torch.dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.factory_kwargs = {'device': self.device, 'dtype': self.dtype}
        self.std = math.sqrt(2/(self.in_features + self.out_features))
        self.weights = nn.Parameter(torch.empty(self.out_features, self.in_features, **self.factory_kwargs))
        torch.nn.init.trunc_normal_(self.weights, mean=0 ,std=self.std, a=-self.std*3, b=self.std*3)

    def forward(self, x: torch.Tensor):
        return einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")