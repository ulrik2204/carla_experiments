from typing import Optional

import cv2
import numpy as np
import torch
from matplotlib import cm
from matplotlib import pyplot as plt
from PIL import Image
from torch.nn.functional import softmax
from torchvision import transforms

from carla_experiments.models.op_deepdive import SequenceBaselineV1

# MED model
MEDMODEL_INPUT_SIZE = (512, 256)
MEDMODEL_YUV_SIZE = (MEDMODEL_INPUT_SIZE[0], MEDMODEL_INPUT_SIZE[1] * 3 // 2)
MEDMODEL_CY = 47.6

medmodel_fl = 910.0
medmodel_intrinsics = np.array(
    [
        [medmodel_fl, 0.0, 0.5 * MEDMODEL_INPUT_SIZE[0]],
        [0.0, medmodel_fl, MEDMODEL_CY],
        [0.0, 0.0, 1.0],
    ]
)
DEVICE_FRAME_FROM_VIEW_FRAME = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
)
VIEW_FRAME_FROM_DEVICE_FRAME = DEVICE_FRAME_FROM_VIEW_FRAME.T
NUM_PTS = 10 * 20  # 10 s * 20 Hz = 200 frames
T_ANCHORS = np.array(
    (
        0.0,
        0.00976562,
        0.0390625,
        0.08789062,
        0.15625,
        0.24414062,
        0.3515625,
        0.47851562,
        0.625,
        0.79101562,
        0.9765625,
        1.18164062,
        1.40625,
        1.65039062,
        1.9140625,
        2.19726562,
        2.5,
        2.82226562,
        3.1640625,
        3.52539062,
        3.90625,
        4.30664062,
        4.7265625,
        5.16601562,
        5.625,
        6.10351562,
        6.6015625,
        7.11914062,
        7.65625,
        8.21289062,
        8.7890625,
        9.38476562,
        10.0,
    )
)
T_IDX = np.linspace(0, 10, num=NUM_PTS)


def calibration(extrinsic_matrix, cam_intrinsics, device_frame_from_road_frame=None):
    if device_frame_from_road_frame is None:
        device_frame_from_road_frame = np.hstack(
            (np.diag([1, -1, -1]), [[0], [0], [1.51]])
        )
    med_frame_from_ground = (
        medmodel_intrinsics
        @ VIEW_FRAME_FROM_DEVICE_FRAME
        @ device_frame_from_road_frame[:, (0, 1, 3)]
    )
    ground_from_med_frame = np.linalg.inv(med_frame_from_ground)

    extrinsic_matrix_eigen = extrinsic_matrix[:3]
    camera_frame_from_road_frame = np.dot(cam_intrinsics, extrinsic_matrix_eigen)
    camera_frame_from_ground = np.zeros((3, 3))
    camera_frame_from_ground[:, 0] = camera_frame_from_road_frame[:, 0]
    camera_frame_from_ground[:, 1] = camera_frame_from_road_frame[:, 1]
    camera_frame_from_ground[:, 2] = camera_frame_from_road_frame[:, 3]
    warp_matrix = np.dot(camera_frame_from_ground, ground_from_med_frame)

    return warp_matrix


def transform_images(
    current_image: Image.Image, last_image: Image.Image, device: str = "cuda"
):
    # seq_input_img
    trans = transforms.Compose(
        [
            # transforms.Resize((900 // 2, 1600 // 2)),
            # transforms.Resize((9 * 32, 16 * 32)),
            transforms.Resize((128, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.3890, 0.3937, 0.3851], [0.2172, 0.2141, 0.2209]),
        ]
    )
    warp_matrix = calibration(
        extrinsic_matrix=np.array(
            [[0, -1, 0, 0], [0, 0, -1, 1.22], [1, 0, 0, 0], [0, 0, 0, 1]]
        ),
        cam_intrinsics=np.array([[910, 0, 582], [0, 910, 437], [0, 0, 1]]),
        device_frame_from_road_frame=np.hstack(
            (np.diag([1, -1, -1]), [[0], [0], [1.22]])
        ),
    )
    imgs = [last_image, current_image]  # contains one more img
    imgs = [
        cv2.warpPerspective(
            src=cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR),
            M=warp_matrix,
            dsize=(512, 256),
            flags=cv2.WARP_INVERSE_MAP,
        )
        for img in imgs
    ]
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
    imgs = list(Image.fromarray(img) for img in imgs)
    imgs = list(trans(img)[None] for img in imgs)
    input_img = torch.cat(imgs, dim=0)  # [N+1, 3, H, W]
    del imgs
    input_img = torch.cat((input_img[:-1, ...], input_img[1:, ...]), dim=1)
    return input_img.to(device)  # Should be [1, 6, 128, 256]


def setup_calling_op_deepdive(
    op_deepdive_model: SequenceBaselineV1,
    batch_size: int = 1,
    device: str = "cuda",
):
    """Sets up calling OP-Deepdive in eval"""
    hidden = torch.zeros((2, batch_size, 512)).to(device)

    def call(input_images: torch.Tensor):
        with torch.no_grad():
            nonlocal hidden
            pred_cls, pred_trajectory, hidden = op_deepdive_model(input_images, hidden)
        pred_conf = softmax(pred_cls, dim=-1)[0]
        # pred_trajectory.Shape = (5, 33, 3)
        pred_trajectory = pred_trajectory.reshape(
            op_deepdive_model.M, op_deepdive_model.num_pts, 3
        )
        return pred_trajectory, pred_conf, pred_cls, hidden

    return call


def opd_input_images_to_rgb(inputs):
    vis_img = (
        inputs.permute(0, 2, 3, 1)[0]
        * torch.tensor((0.2172, 0.2141, 0.2209, 0.2172, 0.2141, 0.2209))
        + torch.tensor((0.3890, 0.3937, 0.3851, 0.3890, 0.3937, 0.3851))
    ) * 255
    vis_img = vis_img.clamp(0, 255)
    previous_image, current_image = vis_img[..., :3].numpy().astype(np.uint8), vis_img[
        ..., 3:
    ].numpy().astype(np.uint8)
    return current_image, previous_image


def plot_trajectory(
    waypoints: np.ndarray,
    show: bool = False,
    save_path: Optional[str] = None,
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Assuming waypoints is a numpy array of shape (33, 3)
    xs = waypoints[:, 0]
    ys = waypoints[:, 1]
    zs = waypoints[:, 2]

    ax.plot(xs, ys, zs, marker=".")

    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    # Save the plot to a file
    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()


def plot_trajectory_with_colors(
    waypoints: np.ndarray,
    show: bool = False,
    save_path: Optional[str] = None,
    xlim=(None, None),
    ylim=(None, None),
    zlim=(None, None),
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Assuming waypoints is a numpy array of shape (n, 3)
    xs = waypoints[:, 0]
    ys = waypoints[:, 1]
    zs = waypoints[:, 2]

    # Compute the number of waypoints
    num_points = len(xs)

    # Create a color map
    cmap = cm.get_cmap("viridis")

    # Plot each segment with a color based on its position in the trajectory
    for i in range(num_points - 1):
        ax.plot(
            xs[i : i + 2],  # noqa
            ys[i : i + 2],  # noqa
            zs[i : i + 2],  # noqa
            color=cmap(i / (num_points - 1)),
            marker=".",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    xmin, xmax = xlim
    ymin, ymax = ylim
    zmin, zmax = zlim
    if xmin and xmax:
        ax.set_xlim(xmin, xmax)
    if ymin and ymax:
        ax.set_ylim(ymin, ymax)
    if zmin and zmax:
        ax.set_zlim(zmin, zmax)

    # Save the plot to a file
    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    # plt.close()


def frd_waypoints_to_fru(waypoints: np.ndarray):
    # Convert from FRD to FRU
    waypoints_fru = waypoints.copy()
    waypoints_fru[:, 2] = -waypoints_fru[:, 2]
    return waypoints_fru
