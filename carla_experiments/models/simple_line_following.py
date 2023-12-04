import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class SimpleLineFollowing(nn.Module):
    def __init__(
        self, output_dim: int, use_weights: bool = True, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        # TODO: Test with and without pretrained weights
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT, **kwargs)
        self.resnet.fc = torch.nn.Identity()  # type: ignore
        self.regressor = nn.Sequential(
            nn.Linear(2048, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        resnet_features = self.resnet(x)
        return self.regressor(resnet_features)
