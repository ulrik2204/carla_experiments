from typing import List, Optional, Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt

from carla_experiments.common.types_common import SupercomboOutputLogged
from carla_experiments.common.utils_op_deepdive import denormalize, img_from_device


# From https://github.com/OpenDriveLab/Openpilot-Deepdive
# Example of usage
# Assuming `processed_array` is your input array of shape (1, 128, 256, 6)
# processed_array = np.random.randn(1, 128, 256, 6).astype(np.float32)  # Dummy array for example
# reconstruct_and_plot_yuv_image(processed_array)
def draw_trajectory_on_ax(
    ax: plt.Axes,
    trajectories: List[np.ndarray],
    confs: List[float],
    line_type="o-",
    transparent=True,
    xlim=(-30, 30),
    ylim=(0, 100),
):
    """
    ax: matplotlib.axes.Axes, the axis to draw trajectories on
    trajectories: List of numpy arrays of shape (num_points, 2 or 3)
    confs: List of numbers, 1 means gt
    """

    # get the max conf
    # print("conf", confs)
    max_conf = max([conf for conf in confs if conf != 1])

    for idx, (trajectory, conf) in enumerate(zip(trajectories, confs)):
        label = "gt" if conf == 1 else "pred%d (%.3f)" % (idx, conf)
        alpha = 1.0
        if transparent:
            alpha = 1.0 if conf == max_conf else np.clip(conf, 0.1, None)
        plot_args = dict(label=label, alpha=alpha, linewidth=2 if alpha == 1.0 else 1)
        if label == "gt":
            plot_args["color"] = "#d62728"
        ax.plot(
            trajectory[:, 1],  # - for nuscenes and + for comma 2k19
            trajectory[:, 0],
            line_type,
            **plot_args,
        )
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend()

    return ax


def draw_path(
    device_path,
    img: np.ndarray,
    width: float = 1,
    height: float = 1.2,
    fill_color: Optional[Tuple[int, int, int]] = (128, 0, 255),
    line_color: Optional[Tuple[int, int, int]] = (0, 255, 0),
    denormalize_imgs: bool = False,
):
    # device_path: N, 3
    device_path_l = device_path + np.array([0, 0, height])
    device_path_r = device_path + np.array([0, 0, height])
    device_path_l[:, 1] -= width
    device_path_r[:, 1] += width

    img_points_norm_l = img_from_device(device_path_l)
    img_points_norm_r = img_from_device(device_path_r)

    img_pts_l = img_points_norm_l
    img_pts_r = img_points_norm_r
    if denormalize_imgs:
        img_pts_l = denormalize(img_pts_l)
        img_pts_r = denormalize(img_pts_r)
    # filter out things rejected along the way
    valid = np.logical_and(
        np.isfinite(img_pts_l).all(axis=1), np.isfinite(img_pts_r).all(axis=1)
    )
    img_pts_l = img_pts_l[valid].astype(int)
    img_pts_r = img_pts_r[valid].astype(int)

    for i in range(1, len(img_pts_l)):
        u1, v1, u2, v2 = np.append(img_pts_l[i - 1], img_pts_r[i - 1])
        u3, v3, u4, v4 = np.append(img_pts_l[i], img_pts_r[i])
        pts = np.array([[u1, v1], [u2, v2], [u4, v4], [u3, v3]], np.int32).reshape(
            (-1, 1, 2)
        )
        if fill_color:
            cv2.fillPoly(img, [pts], fill_color)
        if line_color:
            cv2.polylines(img, [pts], True, line_color)


def visualize_trajectory(
    prev_input_img: np.ndarray,
    current_input_img: np.ndarray,
    pred_outputs: SupercomboOutputLogged,
    gt: SupercomboOutputLogged,
    metric: float,
    save_file: str,
) -> None:
    pred_plan = pred_outputs["plan"]["position"].squeeze()
    # print("pred plan size", pred_plan.shape)
    gt_plan = gt["plan"]["position"].squeeze()
    # pred_conf = pred_outputs["plan"]["position_stds"]

    # Debug prints to check dimensions

    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(16, 9))
    # fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    # fig = plt.figure(figsize=(12, 9.9))  # W, H
    fig = plt.figure(figsize=(12, 9))  # W, H
    spec = fig.add_gridspec(3, 3)  # H, W
    ax1 = fig.add_subplot(spec[2, 0])  # H, W
    ax2 = fig.add_subplot(spec[2, 1])
    ax3 = fig.add_subplot(spec[:, 2])
    ax4 = fig.add_subplot(spec[0:2, 0:2])
    ax1.imshow(prev_input_img)
    ax1.set_title("network input [previous]")
    ax1.axis("off")

    ax2.imshow(current_input_img)
    ax2.set_title("network input [current]")
    ax2.axis("off")

    trajectories = [pred_plan.cpu().numpy(), gt_plan.cpu().numpy()]
    # print("len trajectories", len(trajectories))
    # trajectories = list(labels + 0.8) + list(labels)  # TODO: remove
    confs = [0.5] + [1]
    # confs = [0.5, 1]  # TODO: Remove
    # print("distance", calculate_distances(np.squeeze(labels.numpy())))
    ax3 = draw_trajectory_on_ax(ax3, trajectories, confs, ylim=(0, 200))
    ax3.set_title("Mean L2: %.2f" % metric)
    ax3.grid()

    overlay = current_input_img.copy()
    draw_path(
        pred_plan,
        overlay,
        width=1,
        height=1.2,
        fill_color=(255, 255, 255),
        line_color=(0, 255, 0),
    )
    current_input_img = 0.5 * current_input_img + 0.5 * overlay
    draw_path(
        pred_plan,
        current_input_img,
        width=1,
        height=1.2,
        fill_color=None,
        line_color=(0, 255, 0),
    )

    ax4.imshow(current_input_img.astype(np.uint8))
    ax4.set_title("project on current frame")
    ax4.axis("off")

    plt.tight_layout()
    plt.savefig(
        save_file,
        pad_inches=0.2,
        bbox_inches="tight",
    )
    # plt.show()
    plt.close(fig)
