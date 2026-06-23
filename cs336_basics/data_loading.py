import numpy
import torch
import random

def DataLoading(x: numpy.array, batch_size: int, context_length: int, device: torch.device):

    if len(x) <= context_length + 1:
        raise ValueError(f"Invalid learning rate: {len(x)}, Must be greater then context_length + 1")
    # generate batch_size index
    batch_indices = [random.randint(0, len(x)-context_length-1) for _ in range(batch_size)]

    input_tensor_list = []
    output_tensor_list = []
    for i in range(batch_size):
        input = torch.from_numpy(x[batch_indices[i]: batch_indices[i]+context_length])
        output = torch.from_numpy(x[batch_indices[i]+1: batch_indices[i]+context_length+1])
        input_tensor_list.append(input)
        output_tensor_list.append(output)
    
    return (torch.stack(input_tensor_list).to(device=device), 
            torch.stack(output_tensor_list).to(device=device))

