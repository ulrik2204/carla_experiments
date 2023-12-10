from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, NamedTuple

import click
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, Subset
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from tqdm import tqdm

from carla_experiments.datasets.simple_dataset import SimpleTimeDataset
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
    metrics_dict: Dict[str, float],
    log_to_wandb: bool = True,
):
    print("-- Loss and Metrics --")
    print("Train Loss: ", _format_scores(train_losses_dict))
    print("Val Loss: ", _format_scores(val_losses_dict))
    print("Metrics: ", _format_scores(metrics_dict))
    if log_to_wandb:
        wandb.log(
            {
                "train_loss": train_losses_dict,
                "val_loss": val_losses_dict,
                "metrics": metrics_dict,
            }
        )


class ModelStepResult(NamedTuple):
    steer_loss: torch.Tensor
    throttle_loss: torch.Tensor
    brake_loss: torch.Tensor
    metric_score: np.ndarray


def model_step(
    model: nn.Module,
    image: torch.Tensor,
    control: torch.Tensor,
    loss_criterion: nn.Module,
    metrics: Callable[[torch.Tensor, torch.Tensor], np.ndarray],
    device: str,
):
    image = image.to(device)
    control = control.to(device)
    output = model(image)

    steer_loss, throttle_loss, brake_loss = loss_criterion(output, control)
    metric_score = metrics(output, control)
    return ModelStepResult(steer_loss, throttle_loss, brake_loss, metric_score)


def evaluate_model(
    model: nn.Module,
    loss_criterion: nn.Module,
    metrics: Callable[[torch.Tensor, torch.Tensor], np.ndarray],
    val_dl: DataLoader,
    metric_to_dict: Callable[[np.ndarray], Dict[str, float]],
    device: str,
):
    model.eval()
    len_dl = len(val_dl)
    losses = np.array([0.0, 0.0, 0.0])
    metric_scores = np.zeros(len(metrics(torch.rand([3, 3]), torch.rand([3, 3]))))
    with torch.no_grad():
        for i, (image, control) in (pbar := tqdm(enumerate(val_dl), total=len_dl)):
            steer_loss, throttle_loss, brake_loss, metric_score = model_step(
                model, image, control, loss_criterion, metrics, device
            )
            losses += np.array(
                [steer_loss.item(), throttle_loss.item(), brake_loss.item()]
            )
            metric_scores += metric_score
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
        metric_dict = metric_to_dict(metric_scores / len_dl)
        return val_losses_dict, metric_dict


def train_model(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_criterion: nn.Module,
    metrics: Callable[[torch.Tensor, torch.Tensor], np.ndarray],
    train_dl: DataLoader,
    val_dl: DataLoader,
    metrics_to_dict: Callable[[np.ndarray], Dict[str, float]],
    epochs: int,
    save_folder: Path,
    device: str,
):
    len_dl = len(train_dl)
    for epoch in range(epochs):
        model.train()
        losses = np.array([0.0, 0.0, 0.0])
        print(f"*** Epoch: {epoch} ***")
        for i, (image, control) in (pbar := tqdm(enumerate(train_dl), total=len_dl)):
            steer_loss, throttle_loss, brake_loss, _ = model_step(
                model, image, control, loss_criterion, metrics, device
            )
            losses += np.array(
                [steer_loss.item(), throttle_loss.item(), brake_loss.item()]
            )
            all_loss = steer_loss + throttle_loss + brake_loss
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
        val_losses_dict, metrics_dict = evaluate_model(
            model, loss_criterion, metrics, val_dl, metrics_to_dict, device
        )
        log_scores(train_losses_dict, val_losses_dict, metrics_dict, log_to_wandb=True)
        model_path = (
            save_folder
            / (
                datetime.now().strftime("%d%H%M")
                + f"-loss{train_losses_dict['avg_all']:.4f}.pt"
            )
        ).as_posix()
        save_state_dict(model, optimizer, epoch, model_path)


class LossImpl(nn.Module):
    steer_loss = nn.MSELoss()
    throttle_loss = nn.MSELoss()
    brake_loss = nn.MSELoss()

    def __call__(self, pred, target):
        steer = pred[:, 0]
        throttle = pred[:, 1]
        brake = pred[:, 2]
        target_steer = target[:, 0]
        target_throttle = target[:, 1]
        target_brake = target[:, 2]
        return (
            self.steer_loss(steer, target_steer),
            self.throttle_loss(throttle, target_throttle),
            self.brake_loss(brake, target_brake),
        )

    def to(self, device):
        super().to(device)
        self.steer_loss.to(device)
        self.throttle_loss.to(device)
        self.brake_loss.to(device)
        return self


class Metrics(nn.Module):
    steer_mae = MeanAbsoluteError()
    throttle_mae = MeanAbsoluteError()
    brake_mae = MeanAbsoluteError()

    steer_mse = MeanSquaredError()
    throttle_mse = MeanSquaredError()
    brake_mse = MeanSquaredError()

    def __call__(self, pred, target):
        pred_steer = pred[:, 0]
        pred_throttle = pred[:, 1]
        pred_brake = pred[:, 2]
        target_steer = target[:, 0]
        target_throttle = target[:, 1]
        target_brake = target[:, 2]
        with torch.no_grad():
            steer_mae = self.steer_mae(pred_steer, target_steer)
            throttle_mae = self.throttle_mae(pred_throttle, target_throttle)
            brake_mae = self.brake_mae(pred_brake, target_brake)
            steer_rmse = torch.sqrt(self.steer_mse(pred_steer, target_steer))
            throttle_rmse = torch.sqrt(
                self.throttle_mse(pred_throttle, target_throttle)
            )
            brake_rmse = torch.sqrt(self.brake_mse(pred_brake, target_brake))
            return np.array(
                [
                    steer_mae.item(),
                    throttle_mae.item(),
                    brake_mae.item(),
                    steer_rmse.item(),
                    throttle_rmse.item(),
                    brake_rmse.item(),
                ]
            )

    def to(self, device):
        super().to(device)
        self.steer_mae.to(device)
        self.throttle_mae.to(device)
        self.brake_mae.to(device)
        self.steer_mse.to(device)
        self.throttle_mse.to(device)
        self.brake_mse.to(device)
        return self

    @staticmethod
    def to_dict(metrics: np.ndarray):
        return {
            "steer_mae": float(metrics[0]),
            "throttle_mae": float(metrics[1]),
            "brake_mae": float(metrics[2]),
            "steer_rmse": float(metrics[3]),
            "throttle_rmse": float(metrics[4]),
            "brake_rmse": float(metrics[5]),
        }


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
    # train_transforms = get_simple_time_training_image_transforms()
    # val_test_transforms = get_simple_val_test_transforms()
    all_data = SimpleTimeDataset(dataset_path)
    split_index = int(0.8 * len(all_data))
    train_data = Subset(all_data, range(0, split_index))
    val_data = Subset(all_data, range(split_index, len(all_data)))

    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    model = SimpleLineFollowingB0(output_dim=3, use_pretrained_weights=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_criterion = LossImpl().to(device)
    metrics = Metrics().to(device)
    train_model(
        model,
        optimizer,
        loss_criterion,
        metrics,
        train_dl,
        val_dl,
        Metrics.to_dict,
        epochs,
        save_folder,
        device,
    )


if __name__ == "__main__":
    main()
