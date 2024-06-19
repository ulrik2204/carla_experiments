from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, TypedDict

import torch
import torch.nn as nn
import torch.optim as optim
import wandb


def init_wandb(project_name: str, epochs: int, batch_size: int, learning_rate: float):
    wandb_cache_path = Path("./.wandb")
    time = datetime.now().strftime("%m%d%H%M")
    wandb_cache_path.mkdir(parents=True, exist_ok=True)
    wandb.login()
    wandb.init(
        project=project_name,
        name=f"Experiment {time}",
        dir=wandb_cache_path.as_posix(),
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        },
    )


class ModelCheckpoint(TypedDict):
    epoch: int
    state_dict: Dict[str, torch.Tensor]
    optimizer: Dict[str, torch.Tensor]


def save_state_dict(
    model: nn.Module, optimizer: optim.Optimizer, epoch: int, path: str
):
    checkpoint: ModelCheckpoint = {
        "epoch": epoch + 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)


def load_state_dict(model: nn.Module, optimizer: Optional[optim.Optimizer], path: str):
    checkpoint: ModelCheckpoint = torch.load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer, checkpoint["epoch"] - 1
