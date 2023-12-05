from datetime import datetime
from pathlib import Path

import click
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from carla_experiments.datasets.simple_dataset import (
    SimpleDataset,
    get_training_image_transforms,
)
from carla_experiments.models.simple import SimpleLineFollowing
from carla_experiments.train.training_utils import init_wandb, save_state_dict

DATASET_BASE_PATH = "./output/big_one_dataset/"
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
WEIGHTS_FOLDER = "./.weights/"


def log_scores(
    train_loss: float,
    log_to_wandb: bool = True,
):
    print("-- Loss and Metrics --")
    print("Train Loss: ", train_loss)
    if log_to_wandb:
        wandb.log({"train_loss": train_loss})


def train_model(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_criterion: nn.Module,
    train_dl: DataLoader,
    epochs: int,
    save_folder: Path,
    device: str,
):
    len_dl = len(train_dl)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (image, control) in (pbar := tqdm(enumerate(train_dl), total=len_dl)):
            image = image.to(device)
            control = control.to(device)
            output = model(image)
            loss = loss_criterion(output, control)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            pbar.set_postfix_str(f"TRAIN: avg loss {total_loss/(i+1):.4f}")
        avg_loss = total_loss / len_dl
        log_scores(avg_loss, log_to_wandb=True)
        model_path = (
            save_folder
            / (datetime.now().strftime("%d%H%M") + f"-loss{avg_loss:.4f}.pt")
        ).as_posix()
        save_state_dict(model, optimizer, epoch, model_path)


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
    train_data = SimpleDataset(dataset_path, transform=train_transforms)
    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    model = SimpleLineFollowing(output_dim=3, use_pretrained_weights=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_criterion = torch.nn.MSELoss()
    train_model(
        model,
        optimizer,
        loss_criterion,
        train_dl,
        epochs,
        save_folder,
        device,
    )


if __name__ == "__main__":
    main()
