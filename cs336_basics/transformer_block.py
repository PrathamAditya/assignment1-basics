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
        self.feed_forward =SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.tensor, rope: RotaryPositionalEmbedding = None):
        # print(f"TransformerBlock: {x.shape}")
        token_positions = torch.arange(x.shape[-2], device=x.device)
        y = x + self.attention(self.rms_norm1(x), rope = rope, token_positions=token_positions)
        z = y + self.feed_forward(self.rms_norm2(y))
        return z
# d_model: int Dimensionality of the Transformer block inputs.
# num_heads: int Number of heads to use in multi-head self-attention.
# d_ff: int Dimensionality of the position-wise feed-forward inner layer.