import torch
import torch.nn as nn
import math

class QKNorm(nn.Module):
    """RMSNorm applied independently to each head's Q/K vectors."""
    def __init__(self, head_dim: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.head_dim = head_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(head_dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        result = x / rms * self.weight
        return result.to(in_dtype)