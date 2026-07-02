import torch
import os
import typing


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer
                    , iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    checkpoint_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint_dict, out)


def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                     model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    checkpoint_dict = torch.load(src)
    model.load_state_dict(checkpoint_dict["model"])
    optimizer.load_state_dict(checkpoint_dict["optimizer"])
    start_iteration = checkpoint_dict["iteration"]
    return start_iteration


def save_checkpoint_dual(model, muon_optimizer, adamw_optimizer, iteration, out):
    """
    Same as save_checkpoint, but for the two-optimizer (Muon/NorMuon + AdamW) setup.
    """
    if isinstance(out, (str, os.PathLike)):
        dirname = os.path.dirname(out)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    checkpoint_dict = {
        "model": model.state_dict(),
        "muon_optimizer": muon_optimizer.state_dict(),
        "adamw_optimizer": adamw_optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint_dict, out)


def load_checkpoint_dual(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                          model: torch.nn.Module, muon_optimizer: torch.optim.Optimizer,
                          adamw_optimizer: torch.optim.Optimizer):
    """
    Loads a checkpoint saved with save_checkpoint_dual. Returns start_iteration.
    """
    checkpoint_dict = torch.load(src)
    model.load_state_dict(checkpoint_dict["model"])
    muon_optimizer.load_state_dict(checkpoint_dict["muon_optimizer"])
    adamw_optimizer.load_state_dict(checkpoint_dict["adamw_optimizer"])
    start_iteration = checkpoint_dict["iteration"]
    return start_iteration