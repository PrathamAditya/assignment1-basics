import torch
import torch.nn as nn
import copy
from .rmsnorm import RMSNorm
from .multihead_self_attention import MultiHeadSelfAttention
from .swiglu import SwiGLU
from .transformer_block import TransformerBlock
from .embedding import Embedding
from .linear import Linear
from .softmax import Softmax
from .rope import RotaryPositionalEmbedding

class TransformerLM(nn.Module):
        # final output -> predicted next-token logits
        # no softmax applied
    def _get_clones(self, module, N):
        # Returns N independent deep copies of a single PyTorch module
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
     
    def __init__(self, d_model: int, num_heads: int, d_ff: int, vocab_size: int, context_length: int, 
                  num_layers: int, theta: float, weights: dict[str, torch.Tensor]):
        super().__init__()
        self.base_layer = TransformerBlock(d_model, num_heads, d_ff)
        self.embedding = Embedding(vocab_size, d_model)
        self.embedding.load_state_dict({"weight": weights["token_embeddings.weight"]})
        self.linear = Linear(d_model, vocab_size)
        self.linear.load_state_dict({"weight": weights["lm_head.weight"]})
        # self.linear.load_state_dict({"weights": weights["lm_head.weight"]})
        self.rms_norm_final = RMSNorm(d_model=d_model)
        self.rms_norm_final.load_state_dict({"weight": weights["ln_final.weight"]})
        self.rope = RotaryPositionalEmbedding(theta, int(d_model/num_heads), context_length)
        self.stacked_layers = self._get_clones(self.base_layer, num_layers)
        counter = 0
        for block in self.stacked_layers:
            with torch.no_grad():
                block.rms_norm1.weight.copy_(weights[f"layers.{counter}.ln1.weight"])
                block.rms_norm2.weight.copy_(weights[f"layers.{counter}.ln2.weight"])
                block.attention.W_Q.weight.copy_(weights[f"layers.{counter}.attn.q_proj.weight"])
                block.attention.W_K.weight.copy_(weights[f"layers.{counter}.attn.k_proj.weight"])
                block.attention.W_V.weight.copy_(weights[f"layers.{counter}.attn.v_proj.weight"])
                block.attention.W_O.weight.copy_(weights[f"layers.{counter}.attn.output_proj.weight"])
                block.feed_forward.w1.weight.copy_(weights[f"layers.{counter}.ffn.w1.weight"])
                block.feed_forward.w2.weight.copy_(weights[f"layers.{counter}.ffn.w2.weight"])
                block.feed_forward.w3.weight.copy_(weights[f"layers.{counter}.ffn.w3.weight"])
            counter += 1


    def forward(self, x: torch.Tensor):
        embedding_tensor = self.embedding.forward(x)
        hidden = embedding_tensor

        for block in self.stacked_layers:
            hidden = block(hidden, self.rope)
        
        result = self.rms_norm_final.forward(hidden)
        return self.linear.forward(result)



        

            
