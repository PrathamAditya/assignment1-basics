import torch.nn as nn
import torch
from einops import rearrange
from .scaled_dot_product_attention import ScaledDotProductAttention
from .rope import RotaryPositionalEmbedding
from .rmsnorm import RMSNorm  # adjust path to wherever your RMSNorm actually lives


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, qk_norm: bool = True, eps: float = 1e-5):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = self.d_v = int(d_model / num_heads)
        self.d_model = d_model
        self.W_Q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_O = nn.Linear(self.d_model, self.d_model, bias=False)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.d_k, eps=eps)
            self.k_norm = RMSNorm(self.d_k, eps=eps)

    def forward(self, x: torch.tensor, mask: torch.tensor = None,
                rope: RotaryPositionalEmbedding = None, token_positions=None):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        rearranged_q = rearrange(Q, '... s (h d) -> ... h s d', h=self.num_heads)
        rearranged_k = rearrange(K, '... s (h d) -> ... h s d', h=self.num_heads)
        rearranged_v = rearrange(V, '... s (h d) -> ... h s d', h=self.num_heads)

        if self.qk_norm:
            rearranged_q = self.q_norm(rearranged_q)
            rearranged_k = self.k_norm(rearranged_k)

        if rope is not None:
            rearranged_q = rope(rearranged_q, token_positions)
            rearranged_k = rope(rearranged_k, token_positions)

        if mask is None:
            seq_len = rearranged_q.shape[-2]
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=rearranged_q.device, dtype=torch.bool))
            result = ScaledDotProductAttention(rearranged_q, rearranged_k, rearranged_v, causal_mask)
        else:
            result = ScaledDotProductAttention(rearranged_q, rearranged_k, rearranged_v, mask)

        result = rearrange(result, '... h s d -> ... s (h d)')
        return self.W_O(result)