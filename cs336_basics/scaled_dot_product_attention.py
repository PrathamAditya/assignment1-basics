import torch
import math
from .softmax import Softmax
def ScaledDotProductAttention(Q: torch.Tensor, K: torch.Tensor, V :torch.Tensor, mask: torch.Tensor = None):

    k_t = torch.transpose(K, -1, -2)
    scores = Q @ k_t
    dk = Q.shape[-1]
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    scores_scaled = scores/math.sqrt(dk)

    return Softmax(scores_scaled, -1) @ V
