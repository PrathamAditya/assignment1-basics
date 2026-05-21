import torch.nn as nn
import torch

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        bands = torch.arange(0, self.d_k, 2, device=self.device)[: (self.d_k // 2)].float()
        inv_freq = 1.0 / (self.theta ** (bands/ self.d_k))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(self.max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, num_heads, seq_len, d_k)
        H, S, D = x.shape

        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        
        if token_positions.dim() == 1:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
            
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        x_paired = x.view(H, S, D // 2, 2)
        
        x0 = x_paired[..., 0] 
        x1 = x_paired[..., 1]
        
        x0_rotated = x0 * cos - x1 * sin
        x1_rotated = x0 * sin + x1 * cos
        
        out_paired = torch.stack([x0_rotated, x1_rotated], dim=-1)

        return out_paired.view(H, S, D)