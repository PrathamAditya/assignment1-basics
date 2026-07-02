import numpy
import torch
import random
import numpy as np

def DataLoading(x: numpy.array, batch_size: int, context_length: int, device: torch.device):

    if len(x) <= context_length + 1:
        raise ValueError(f"Invalid learning rate: {len(x)}, Must be greater then context_length + 1")
    # batch_indices = [random.randint(0, len(x)-context_length-1) for _ in range(batch_size)]
    batch_indices = np.random.randint(
    0,
    len(x) - context_length - 1,
    size=batch_size,
)
    offsets = np.arange(context_length)

    inputs = x[batch_indices[:, None] + offsets]
    targets = x[batch_indices[:, None] + offsets + 1]

    inputs = torch.from_numpy(inputs).long()
    targets = torch.from_numpy(targets).long()
    # input_tensor_list = []
    # output_tensor_list = []
    # for i in range(batch_size):
    #     input = torch.from_numpy(x[batch_indices[i]: batch_indices[i]+context_length]).long()
    #     output = torch.from_numpy(x[batch_indices[i]+1: batch_indices[i]+context_length+1]).long()
    #     input_tensor_list.append(input)
    #     output_tensor_list.append(output)
    return (
    inputs.pin_memory().to(device, non_blocking=True),
    targets.pin_memory().to(device, non_blocking=True),
)
    # return (torch.stack(input_tensor_list).to(device=device), 
    #         torch.stack(output_tensor_list).to(device=device))
#     return (
#     torch.stack(input_tensor_list)
#         .pin_memory()
#         .to(device, non_blocking=True),

#     torch.stack(output_tensor_list)
#         .pin_memory()
#         .to(device, non_blocking=True),
# )

def DataLoadingValidationSequential(x: np.ndarray, batch_size: int, context_length: int, device: torch.device):
    """
    Sequentially yields batches across the entire validation dataset.
    Ensures every token is evaluated exactly once.
    """
    # Number of tokens consumed by a single batch step
    tokens_per_batch = batch_size * context_length
    
    # Total tokens available to safely make full contexts + 1 target token
    total_tokens = len(x)
    
    # Loop sequentially through the dataset
    for i in range(0, total_tokens - context_length - 1, tokens_per_batch):
        
        input_tensor_list = []
        output_tensor_list = []
        
        # Build the batch sequentially
        for b in range(batch_size):
            # Calculate the exact start index for this batch element
            start_idx = i + (b * context_length)
            
            # Ensure we don't read past the end of the array
            if start_idx + context_length + 1 > total_tokens:
                break
                
            input_seq = torch.from_numpy(x[start_idx : start_idx + context_length]).long()
            output_seq = torch.from_numpy(x[start_idx + 1 : start_idx + context_length + 1]).long()
            
            input_tensor_list.append(input_seq)
            output_tensor_list.append(output_seq)
        
        # If we reached the very end and couldn't fill a full batch, 
        # we can still yield what we have left
        if len(input_tensor_list) == 0:
            break
            
        yield (torch.stack(input_tensor_list).to(device), 
               torch.stack(output_tensor_list).to(device))
