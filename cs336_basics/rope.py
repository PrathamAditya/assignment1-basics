import torch.nn as nn
import torch
from einops import rearrange

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        bands = torch.arange(0, self.d_k, 2, device=self.device)[: (self.d_k // 2)].float()
        # bands = torch.arange(0, self.d_k, 2, device=self.device).float()
        inv_freq = 1.0 / (self.theta ** (bands/ self.d_k))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(self.max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # print(token_positions.shape)
        # print(token_positions)
        
        cos = self.cos_cached[token_positions]  # Shape: (B, S, D // 2)
        sin = self.sin_cached[token_positions]
        cos = rearrange(cos, "... s d -> ... 1 s d")
        sin = rearrange(sin, "... s d -> ... 1 s d")
        x_paired = rearrange(x, "... s (d pair) -> ... s d pair", pair=2)
        # print(f"ROPE: {x.shape}")
        # print(f"ROPE: {cos.shape}")

        
        x0 = x_paired[..., 0] 
        x1 = x_paired[..., 1]
        # print(x0.shape)
        # print(cos.shape)
        
        x0_rotated = x0 * cos - x1 * sin
        x1_rotated = x0 * sin + x1 * cos
        
        out_paired = torch.stack([x0_rotated, x1_rotated], dim=-1)

        return rearrange(out_paired, "... s d pair -> ... s (d pair)")