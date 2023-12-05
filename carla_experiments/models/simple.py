import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

trans = ResNet50_Weights.IMAGENET1K_V2.transforms


class SimpleLineFollowing(nn.Module):
    def __init__(
        self, output_dim: int, use_pretrained_weights: bool = True, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        pretrained_weights = (
            ResNet50_Weights.IMAGENET1K_V2 if use_pretrained_weights else None
        )
        # Torchvision's resnet50 transforms the input images to 3x224x224
        self.resnet = resnet50(weights=pretrained_weights, **kwargs)
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
