import torch
# ----------------------
# SHAPE AND BROADCASTING
# ----------------------

# BASIC TENSOR
x = torch.randn(2, 3)
y = torch.randn(3)

## broadcasting rule: RIGHT → LEFT, dimensions must either match OR one must be 1.
# (32,256,512) + (512,)
# 512 vs 512
# 256 vs missing -> 1
# 32 vs missing -> 1

#(4,1,8) and (3,8): works!
# print(x)
# print(y)
# print(x+y)

# unsqueeze
# a = torch.randn(256)
# print(a.shape)
# a.unsqueeze(0).unsqueeze(0)  # This is EXACTLY what happens constantly in transformer code.
# print(a.shape)

# Dimension Meaning
# (32,256,512): batch = 32, seq = 256, embedding = 512
#--------------------------------------------------------------------------------------------------------------

# TRANSPOSE/RESHAPE/VIEW
# Example: Attention tensor: (batch, seq, heads, d_k)
# Often transformed into: (batch, heads, seq, d_k)

# a = torch.randn(3,2,10)
# print(a)
# print(a.shape)
# a = a.transpose(0,2)
# print(a.shape)
# a = a.reshape(12, 5)
# a = a.reshape(12, -1) automatic inference.
# print(a.shape)
# tipical in transformers
# x = x.transpose(...).contiguous().view(...)
# .view(): similar to reshape but required contigious memory and does not copy data.

# BATCHED MATMUL
# a = torch.randn(9,2,3)
# b = torch.randn(1,3,4)

# c = a @ b
# # (batch, heads, seq_len, d_k) @ (batch, heads, d_k, seq_len)
# print(c.shape)

# MASKING AND INDEXING
# Basic indexing
# a = torch.tensor([[1,2,3], [4,5,6]])
# print(a[1])

# Row / column access
# print(a[1, 2])

# Slicing
# print(a[1: 2])

# Boolean masking
# mask = a > 3
# print(mask)

# Direct masking
# a = torch.tensor([[1,2,3,20,25], [4,5,6, 21, 44]])
# print(a[a % 2 == 0])
# print(a)

# Assignment with masking
# a = torch.tensor([[1,2,3,20,25], [4,5,6, 21, 44]])
# a = torch.tensor([1,2,3,4,5])
# a[a > 3] = 0
# print(a)

# Masking in transformers
# Very important for attention.
# Example attention scores:
# scores.shape = (batch, seq, seq)
# Mask future tokens:
# scores = scores.masked_fill(mask == 0, -1e9)
# Then softmax makes masked positions almost zero probability.

# Fancy indexing
# a = torch.tensor([10,20,30,40])
# idx = torch.tensor([0,2])
# print(a[idx])

# Multi-dimensional masking
# a = torch.randn(3,3)
# print(a[a > 0]) # return result in 1D

# Dtype Why dtype matters
# It affects:
# memory usage
# speed
# numerical precision
# numerical stability

# AUTOGRAD 
# x = torch.tensor(3.0, requires_grad=True)
# y = x**2
# print(y.backward())

# DEVICE HANDLING
# print(torch.cuda.is_available())
# device = torch.device("cuda")

# x = torch.randn(3,4)
# # Inputs and model must be on same device
# x = x.to(device)
# model = model.to(device)

a = torch.randn(3, device="cuda")
b = torch.randn(3, device="cuda")
print(a + b)