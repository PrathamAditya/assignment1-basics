import torch
import os
import typing
def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer
                    , iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    
    checkpoint_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint_dict, out)
    

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO|typing.IO[bytes], 
                    model: torch.nn.Module, optimizer: torch.optim.Optimizer):

    checkpoint_dict = torch.load(src)
    model.load_state_dict(checkpoint_dict["model"])
    optimizer.load_state_dict(checkpoint_dict["optimizer"])
    start_iteration = checkpoint_dict["iteration"]
    
    return start_iteration
