
import torch

def GradientClipping(params: list, max_l2_norm):
    long_num = 0.0
    scale_down_factor = 1 
    for p in params:
        if p.grad is not None:
            long_num += torch.sum(p.grad ** 2) 

    global_norm = torch.sqrt(long_num)
    if global_norm > max_l2_norm:
        scale_down_factor = max_l2_norm/(global_norm + 1e-6)

    for p in params:
        if p.grad is not None:
            p.grad = p.grad.mul_(scale_down_factor)
