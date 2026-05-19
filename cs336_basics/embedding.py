
import torch.nn as nn
import torch

class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.dtype=None, dtype: torch.dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.factory_kwargs = {'device': self.device, 'dtype': self.dtype}
        self.weight = nn.Parameter(torch.empty(self.num_embeddings, self.embedding_dim, **self.factory_kwargs))
        torch.nn.init.trunc_normal_(self.weight, mean=0 , std=1, a=-3, b=3)
    
    def forward(self, token_ids: torch.Tensor):
        return self.weight[token_ids]
