import torch.nn as nn
import torch
import einops
from .rmsnorm import RMSNorm
from .multihead_self_attention import MultiHeadSelfAttention
from .swiglu import SwiGLU
from .rope import RotaryPositionalEmbedding


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.rms_norm1 = RMSNorm(d_model=d_model)
        self.attention = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
        self.rms_norm2 = RMSNorm(d_model=d_model)
        self.feed_forward = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor, rope: RotaryPositionalEmbedding = None):

        token_positions = torch.arange(x.shape[-2], device=x.device)
        y = x + self.attention(self.rms_norm1(x), rope = rope, token_positions=token_positions)
        z = y + self.feed_forward(self.rms_norm2(y))
        return z