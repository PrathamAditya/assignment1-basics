import torch

def Softmax(input: torch.tensor, dim: int):
    if dim > input.ndim:
        return input    

    max_values, _ = torch.max(input, dim=dim, keepdim=True)
    stable_input = input - max_values
    exp_tensor = torch.exp(stable_input)
    sum_exp = torch.sum(exp_tensor, dim=dim, keepdim=True)
    return exp_tensor / sum_exp
    