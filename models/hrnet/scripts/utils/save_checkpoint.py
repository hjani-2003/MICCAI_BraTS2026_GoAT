import os
import torch
import torch.nn as nn


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, logger, epoch: int, path: str, best_acc=0, scheduler=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_acc": best_acc,
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, path)
    logger.info(f"Checkpoint saved at {path}")
