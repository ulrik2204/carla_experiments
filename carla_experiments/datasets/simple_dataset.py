import json
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(
        self,
        path: str,
        transform: Optional[Callable[[Any], Any]] = None,
        target_transform: Optional[Callable[[Any], Any]] = None,
    ):
        """Dataset class for a simple line following dataset in CARLA.
        The takes in a path to a folder containing two subfolders: images and controls.
        Here the images folder is a folder or images on the form id.png or id.jpg.
        The controls folder is a folder containing json files on the form id.json.
        The order of the images in the images folder and the json files in the controls folder matters.
        When iterating over the dataset it returns a tuple of the PIL image and the corresponding
        control tensor in the format [steer, throttle, brake].

        Args:
            path (str): The path to a folder containing two subfolders: images and controls.
            transform (Callable[[Any], Any]): Transforms om the PIL image.
            target_transform (Callable[[Any], Any]): Transforms on the control tensor.

        Raises:
            ValueError: If the number of images and controls do not match.
        """
        self.transform = transform
        self.target_transform = target_transform
        path_used = Path(path)
        images_folder = path_used / "images"
        controls_folder = path_used / "controls"
        self.images_arr = []
        for image_path in images_folder.iterdir():
            self.images_arr.append(image_path.as_posix())

        control_paths = []
        for control_path in controls_folder.iterdir():
            control_paths.append(control_path.as_posix())

        if len(self.images_arr) != len(control_paths):
            raise ValueError("Number of images and controls do not match")

        # Load control paths
        self.control_tensors = []
        for control_path in control_paths:
            with open(control_path, "r") as f:
                control_dict = json.load(f)
                control_tensor = torch.tensor(
                    [
                        control_dict["steer"],
                        control_dict["throttle"],
                        control_dict["brake"],
                    ]
                )
                self.control_tensors.append(control_tensor)

    def __len__(self):
        return len(self.images_arr)

    def __getitem__(self, idx):
        image = Image.open(self.images_arr[idx])
        control_tensor = self.control_tensors[idx]
        transformed_image = self.transform(image) if self.transform else image
        transformed_target = (
            self.target_transform(control_tensor)
            if self.target_transform
            else control_tensor
        )
        return transformed_image, transformed_target


def get_training_image_transforms():
    return transforms.Compose(
        [
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
        ]
    )


def get_val_test_transforms():
    return transforms.Compose(
        [
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
        ]
    )
