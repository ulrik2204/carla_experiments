from datetime import datetime
from pathlib import Path
from typing import Dict

import click
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from carla_experiments.datasets.simple_dataset import (
    SimpleDataset,
    get_training_image_transforms,
)
from carla_experiments.models.simple import SimpleLineFollowingB0
from carla_experiments.train.training_utils import init_wandb, save_state_dict

DATASET_BASE_PATH = "./output/big_one_dataset/"
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
WEIGHTS_FOLDER = "./.weights/agentB0"


def _format_scores(scores_dict: Dict[str, float]) -> str:
    return "; ".join(map(lambda item: f"{item[0]}: {item[1]:.4f}", scores_dict.items()))


def log_scores(
    train_losses_dict: Dict[str, float],
    val_losses_dict: Dict[str, float],
    log_to_wandb: bool = True,
):
    print("-- Loss and Metrics --")
    print("Train Loss: ", _format_scores(train_losses_dict))
    print("Val Loss: ", _format_scores(val_losses_dict))
    if log_to_wandb:
        wandb.log({"train_loss": train_losses_dict, "val_loss": val_losses_dict})


def evaluate_model(
    model: nn.Module, loss_criterion: nn.Module, val_dl: DataLoader, device: str
):
    model.eval()
    len_dl = len(val_dl)
    losses = np.array([0.0, 0.0, 0.0])
    with torch.no_grad():
        for i, (image, control) in (pbar := tqdm(enumerate(val_dl), total=len_dl)):
            image = image.to(device)
            control = control.to(device)
            output = model(image)

            steer_loss, throttle_loss, brake_loss = loss_criterion(output, control)
            losses += np.array(
                [steer_loss.item(), throttle_loss.item(), brake_loss.item()]
            )
            average_losses = losses / (i + 1)
            pbar.set_postfix_str(
                f"VAL: avg all loss {np.average(average_losses):.6f}, avg losses: steer {average_losses[0]:.6f}; "
                + f"throttle {average_losses[1]:.6f}; brake {average_losses[2]:.6f}"
            )
        val_losses_dict = {
            "avg_all": np.average(losses / len_dl),
            "steer": losses[0] / len_dl,
            "throttle": losses[1] / len_dl,
            "brake": losses[2] / len_dl,
        }
        return val_losses_dict


def train_model(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_criterion: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    epochs: int,
    save_folder: Path,
    device: str,
):
    len_dl = len(train_dl)
    for epoch in range(epochs):
        model.train()
        losses = np.array([0.0, 0.0, 0.0])
        for i, (image, control) in (pbar := tqdm(enumerate(train_dl), total=len_dl)):
            image = image.to(device)
            control = control.to(device)
            output = model(image)

            steer_loss, throttle_loss, brake_loss = loss_criterion(output, control)
            all_loss = steer_loss + throttle_loss + brake_loss
            losses += np.array(
                [steer_loss.item(), throttle_loss.item(), brake_loss.item()]
            )
            all_loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            average_losses = losses / (i + 1)
            pbar.set_postfix_str(
                f"TRAIN: avg all loss {np.average(average_losses):.6f}, avg losses: steer {average_losses[0]:.6f}; "
                + f"throttle {average_losses[1]:.6f}; brake {average_losses[2]:.6f}"
            )
        train_losses_dict = {
            "avg_all": np.average(losses / len_dl),
            "steer": losses[0] / len_dl,
            "throttle": losses[1] / len_dl,
            "brake": losses[2] / len_dl,
        }
        val_losses_dict = evaluate_model(model, loss_criterion, val_dl, device)
        log_scores(train_losses_dict, val_losses_dict, log_to_wandb=True)
        model_path = (
            save_folder
            / (
                datetime.now().strftime("%d%H%M")
                + f"-loss{train_losses_dict['avg_all']:.4f}.pt"
            )
        ).as_posix()
        save_state_dict(model, optimizer, epoch, model_path)


class LossImpl(nn.Module):
    steerLoss = nn.L1Loss()
    throttleLoss = nn.L1Loss()
    brakeLoss = nn.L1Loss()

    def __call__(self, output, control):
        steer = output[:, 0]
        throttle = output[:, 1]
        brake = output[:, 2]
        steer_gt = control[:, 0]
        throttle_gt = control[:, 1]
        brake_gt = control[:, 2]
        return (
            self.steerLoss(steer, steer_gt),
            self.throttleLoss(throttle, throttle_gt),
            self.brakeLoss(brake, brake_gt),
        )


@click.command()
@click.option(
    "--dataset-path", default=DATASET_BASE_PATH, help="Path to the Cityscapes dataset"
)
@click.option("--epochs", default=EPOCHS, help="Amount of epochs to train")
@click.option(
    "--batch-size",
    default=BATCH_SIZE,
    help="Batch size for each training step",
)
@click.option("--learning-rate", default=LEARNING_RATE, help="Learning rate")
@click.option("--device", default="cuda", help="Device to train on")
@click.option(
    "--weights-folder", default=WEIGHTS_FOLDER, help="Folder to save weights in"
)
@click.option(
    "--resume-from-weights",
    default=None,
    help="Path to weights to resume training on. If none, starts from scratch.",
)
def main(
    dataset_path: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    weights_folder: str,
    resume_from_weights: str,
):
    print(
        f"""Args:
        dataset_path={dataset_path},
        epochs={epochs},
        batch_size={batch_size},
        learning_rate={learning_rate},
        device={device},
        weights_folder={weights_folder},
        resume_from_weights={resume_from_weights}
        """
    )
    device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    init_wandb(
        "Training simple line following model", epochs, batch_size, learning_rate
    )
    save_folder = Path(weights_folder)
    save_folder.mkdir(exist_ok=True)
    train_transforms = get_training_image_transforms()
    # val_test_transforms = get_val_test_transforms()
    all_data = SimpleDataset(dataset_path, transform=train_transforms)
    split_index = int(0.8 * len(all_data))
    train_data = Subset(all_data, range(0, split_index))
    val_data = Subset(all_data, range(split_index, len(all_data)))

    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    model = SimpleLineFollowingB0(output_dim=3, use_pretrained_weights=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_criterion = LossImpl()
    train_model(
        model,
        optimizer,
        loss_criterion,
        train_dl,
        val_dl,
        epochs,
        save_folder,
        device,
    )


if __name__ == "__main__":
    main()
