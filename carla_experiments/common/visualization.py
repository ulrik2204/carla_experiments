from pathlib import Path
from typing import List, Optional, Sequence, Tuple, TypedDict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from torch import fill

from carla_experiments.common.types_common import (
    SupercomboFullOutput,
    SupercomboOutputLogged,
)
from carla_experiments.common.utils_op_deepdive import denormalize, img_from_device
from carla_experiments.common.utils_openpilot import (
    car_space_to_bb,
    get_view_frame_from_calib_frame,
    tici_ecam_intrinsics,
    tici_fcam_intrinsics,
)
from carla_experiments.datasets.comma3x_dataset import get_dict_shape


# From https://github.com/OpenDriveLab/Openpilot-Deepdive
# Example of usage
# Assuming `processed_array` is your input array of shape (1, 128, 256, 6)
# processed_array = np.random.randn(1, 128, 256, 6).astype(np.float32)  # Dummy array for example
# reconstruct_and_plot_yuv_image(processed_array)
def draw_trajectory_on_ax(
    ax: plt.Axes,
    trajectories: List[np.ndarray],
    names: List[str],
    confs: List[float],
    pred_lead: List[Tuple[torch.Tensor, float]] = list(),
    gt_lead: List[Tuple[torch.Tensor, float]] = list(),
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

    # dra leads
    for i, (path, prob) in enumerate(pred_lead):
        path = path.cpu().numpy()
        x, y = path[0][:2]
        color = (0, int(255 * prob) / 255, 0)
        ax.plot(
            path[:, 1],
            path[:, 0],
            "o-",
            color=color,
            alpha=prob,
            label=f"pred_lead{i}@{prob}",
            linewidth=2,
        )
    for i, (path, prob) in enumerate(gt_lead):
        path = path.cpu().numpy()
        color = (0, int(255 * prob) / 255, 0)
        ax.plot(
            path[:, 1],
            path[:, 0],
            "o-",
            color=color,
            alpha=prob,
            label=f"gt_lead{i}@{prob}",
            linewidth=2,
        )

    for idx, (trajectory, conf, label) in enumerate(zip(trajectories, confs, names)):
        alpha = 1.0
        if transparent:
            alpha = 1.0 if conf == max_conf else np.clip(conf, 0.1, None)
        plot_args = dict(label=label, alpha=alpha, linewidth=2 if alpha == 1.0 else 1)
        if "original_plan" in label:
            plot_args["color"] = "#d62728"
        if "pred_plan" in label:
            plot_args["color"] = "#1f77b4"
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


def draw_lead(
    lead_output,
    img,
    height=0.5,
    line_color=(0, 255, 0),
    intrinsics: Optional[np.ndarray] = None,
    rpy: Optional[np.ndarray] = np.array([0, 0, 0]),
):
    width = 0.8
    # Assuming lead_output is a torch tensor
    # lead_output = F.pad(lead_output[:, :2], (0, 1))[0].unsqueeze(0).numpy()

    # Define the four corners of the box
    device_path_tl = lead_output + np.array([0, -width / 2, height])  # Top-left
    device_path_tr = lead_output + np.array([0, width / 2, height])  # Top-right
    device_path_bl = lead_output + np.array([0, -width / 2, 0])  # Bottom-left
    device_path_br = lead_output + np.array([0, width / 2, 0])  # Bottom-right

    # Convert these points to image coordinates
    img_pts_tl = car_space_to_bb(
        device_path_tl[:, 0],
        device_path_tl[:, 1],
        device_path_tl[:, 2],
        intrinsic=intrinsics,
        rpy=rpy,
    )
    img_pts_tr = car_space_to_bb(
        device_path_tr[:, 0],
        device_path_tr[:, 1],
        device_path_tr[:, 2],
        intrinsic=intrinsics,
        rpy=rpy,
    )
    img_pts_bl = car_space_to_bb(
        device_path_bl[:, 0],
        device_path_bl[:, 1],
        device_path_bl[:, 2],
        intrinsic=intrinsics,
        rpy=rpy,
    )
    img_pts_br = car_space_to_bb(
        device_path_br[:, 0],
        device_path_br[:, 1],
        device_path_br[:, 2],
        intrinsic=intrinsics,
        rpy=rpy,
    )

    # Filter out invalid points
    valid = np.logical_and.reduce(
        (
            np.isfinite(img_pts_tl).all(axis=1),
            np.isfinite(img_pts_tr).all(axis=1),
            np.isfinite(img_pts_bl).all(axis=1),
            np.isfinite(img_pts_br).all(axis=1),
        )
    )
    img_pts_tl = img_pts_tl[valid].astype(int)
    img_pts_tr = img_pts_tr[valid].astype(int)
    img_pts_bl = img_pts_bl[valid].astype(int)
    img_pts_br = img_pts_br[valid].astype(int)

    for i in range(len(img_pts_tl)):
        pts = np.array(
            [
                [img_pts_tl[i, 0], img_pts_tl[i, 1]],
                [img_pts_tr[i, 0], img_pts_tr[i, 1]],
                [img_pts_br[i, 0], img_pts_br[i, 1]],
                [img_pts_bl[i, 0], img_pts_bl[i, 1]],
            ],
            np.int32,
        ).reshape((-1, 1, 2))

        cv2.polylines(img, [pts], True, line_color, thickness=2)


def draw_line(
    device_path,
    img,
    height: float = 0,
    line_color=(0, 255, 0),
    intrinsics: Optional[np.ndarray] = None,
    rpy: Optional[np.ndarray] = np.array([0, 0, 0]),
):

    # device_path: N, 3
    device_path_l = device_path + np.array([0, 0, height])
    intrinsics = tici_fcam_intrinsics if intrinsics is None else intrinsics

    img_points_norm_l = car_space_to_bb(
        device_path_l[:, 0],
        device_path_l[:, 1],
        device_path_l[:, 2],
        rpy=rpy,
        intrinsic=intrinsics,
    )

    img_pts_l = img_points_norm_l  # denormalize(img_points_norm_l, intrinsics=intrinsics, height=h, width=w)
    # filter out things rejected along the way
    valid = np.isfinite(img_pts_l).all(axis=1)
    img_pts_l = img_pts_l[valid].astype(int)

    for i in range(1, len(img_pts_l)):
        u1, v1 = img_pts_l[i - 1]
        u2, v2 = img_pts_l[i]
        if line_color:
            cv2.line(img, (u1, v1), (u2, v2), line_color, thickness=2)


def draw_path(
    device_path,
    img,
    width=1.0,
    height=1.2,
    fill_color=(128, 0, 255),
    intrinsics: Optional[np.ndarray] = None,
    line_color=(0, 255, 0),
    rpy: Optional[np.ndarray] = np.array([0, 0, 0]),
):
    # device_path: N, 3
    device_path_l = device_path + np.array([0, 0, height])
    device_path_r = device_path + np.array([0, 0, height])
    device_path_l[:, 1] -= width
    device_path_r[:, 1] += width

    # img_points_norm_l = img_from_device(device_path_l)
    # img_points_norm_r = img_from_device(device_path_r)
    # h = img.shape[0]
    # w = img.shape[1]

    # img_pts_l = denormalize(img_points_norm_l, intrinsics=intrinsics, height=h, width=w)
    # img_pts_r = denormalize(img_points_norm_r, intrinsics=intrinsics, height=h, width=w)
    img_pts_l = car_space_to_bb(
        device_path_l[:, 0],
        device_path_l[:, 1],
        device_path_l[:, 2],
        rpy=rpy,
        intrinsic=intrinsics,
    )
    img_pts_r = car_space_to_bb(
        device_path_r[:, 0],
        device_path_r[:, 1],
        device_path_r[:, 2],
        rpy=rpy,
        intrinsic=intrinsics,
    )
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


def resize_and_pad(image, target_width, target_height):
    # Load the image

    # Get the dimensions of the image
    original_height, original_width = image.shape[:2]

    # Calculate the scaling factor while keeping the aspect ratio
    scaling_factor = min(target_width / original_width, target_height / original_height)

    # Resize the image with the scaling factor
    new_width = int(original_width * scaling_factor)
    new_height = int(original_height * scaling_factor)
    resized_image = cv2.resize(image, (new_width, new_height))

    # Create a new image with the target size and fill it with black pixels
    padded_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Calculate the position to place the resized image at the bottom
    y_offset = target_height - new_height
    x_offset = (target_width - new_width) // 2

    # Place the resized image in the padded image
    padded_image[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = (
        resized_image
    )

    return padded_image


Leads = List[Tuple[torch.Tensor, float]]


def plot_leading_vehicle_on_ax(ax: Axes, pred_lead: Leads, gt_lead: Leads):
    # lead is a tensor of shape (33, 6, 4), that is the (x, y, v, a)
    # for 6 points into the future for 3 leading vehicles
    for i, (path, prob) in enumerate(pred_lead):
        path = path.cpu().numpy()
        if prob < 0.5:
            continue
        x, y = path[0][:2]
        color = (0, int(255 * prob) / 255, 0)
        ax.plot(
            path[:, 1],
            path[:, 0],
            "o-",
            color=color,
            alpha=prob,
            label=f"pred_lead{i}@{prob}",
        )
    for i, (path, prob) in enumerate(gt_lead):
        path = path.cpu().numpy()
        if prob < 0.5:
            continue
        color = (0, int(255 * prob) / 255, 0)
        ax.plot(
            path[:, 1],
            path[:, 0],
            "o-",
            color=color,
            alpha=prob,
            label=f"gt_lead{i}@{prob}",
        )


def create_plots(
    image: np.ndarray,
    trajectories: List[np.ndarray],
    labels: List[str],
    confs: List[float],
    colors: Sequence[Tuple[int, int, int]],
    height: float,
    rpy: np.ndarray,
    save_path: str,
    title: str,
    draw_on_img_func=draw_line,
):
    image = image.copy()
    fig = plt.figure(figsize=(12, 9))  # W, H
    spec = fig.add_gridspec(3, 3)  # H, W
    # ax1 = fig.add_subplot(spec[2, 0])  # H, W
    # ax2 = fig.add_subplot(spec[2, 1])
    ax3 = fig.add_subplot(spec[:, 2])
    ax4 = fig.add_subplot(spec[0:2, 0:2])
    ax3 = draw_trajectory_on_ax(
        ax3,
        trajectories,
        labels,
        confs,
        xlim=(-10, 10),
        ylim=(0, 200),
    )
    for trajectory, color in zip(trajectories, colors):
        draw_on_img_func(
            trajectory,
            image,
            height=height,
            line_color=color,
            intrinsics=tici_fcam_intrinsics,
            rpy=rpy,
        )

    ax4.imshow(image.astype(np.uint8))
    ax4.set_title("Project on current frame")
    ax4.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(
        save_path,
        pad_inches=0.2,
        bbox_inches="tight",
    )
    # plt.show()
    plt.close(fig)


def plot_driving_trajectory(
    original_img: np.ndarray,
    prev_input_img: np.ndarray,
    current_input_img: np.ndarray,
    pred_trajectory: np.ndarray,
    gt_trajectory: np.ndarray,
    pred_conf: float,
    rpy_calib: np.ndarray,
    plot_title: str,
    save_file: str,
):
    # image = image.copy()
    # size = (1928, 1208)
    # original_img = resize_and_pad(image, size[0], size[1])

    pred_plan = pred_trajectory
    plan_conf = pred_conf
    height = 1.2
    width = 1.0
    # height = gt["road_transform"][0, 2].item()
    # print("height", height)
    # print("pred plan size", pred_plan.shape)
    gt_plan = gt_trajectory
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

    trajectories = [pred_plan, gt_plan]
    confs = [plan_conf] + [1]
    names = [f"pred_plan@{plan_conf}", "original_plan"]

    ax3 = draw_trajectory_on_ax(
        ax3, trajectories, names, confs, ylim=(0, 200), xlim=(-10, 10)
    )

    ax3.set_title(plot_title)
    ax3.grid()

    overlay = original_img.copy()
    norm_trajectory = trajectories[0]
    draw_path(
        norm_trajectory,
        overlay,
        width=width,
        height=height,
        fill_color=(255, 255, 255),
        line_color=(0, 255, 0),
        intrinsics=tici_fcam_intrinsics,
        rpy=rpy_calib,
    )
    original_img = 0.5 * original_img + 0.5 * overlay
    # draw_path(
    #     norm_trajectory,
    #     original_img,
    #     width=width,
    #     height=height,
    #     fill_color=None,
    #     line_color=(0, 255, 0),
    #     intrinsics=tici_fcam_intrinsics,
    #     rpy=rpy_calib,
    # )
    # print(get_dict_shape(pred_outputs))

    ax4.imshow(original_img.astype(np.uint8))
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


def to_3d(traj_2d: torch.Tensor) -> List[np.ndarray]:
    return [F.pad(path, (0, 1)).cpu().numpy() for path in traj_2d]


def stds_to_probs(stds: List[float]) -> List[float]:
    return [max(1 - s, 0) for s in stds]


class ModelOutputMetrics(TypedDict):
    lane_lines_l2: float
    road_edges_l2: float
    lead_l2: float
    plan_l2: float


def visualize_trajectory(
    original_img: np.ndarray,
    prev_input_img: np.ndarray,
    current_input_img: np.ndarray,
    pred_outputs: SupercomboFullOutput,
    gt: SupercomboOutputLogged,
    rpy_calib: np.ndarray,
    base_path: Path,
    image_name: str,
    model_output_metrics: ModelOutputMetrics,
):

    size = (1928, 1208)
    original_img = resize_and_pad(original_img, size[0], size[1])
    # Plotting lane lines
    lane_lines_base_path = base_path / "lane_lines"
    lane_lines_base_path.mkdir(parents=True, exist_ok=True)
    pred_lane_lines = to_3d(pred_outputs["lane_lines"][0])
    pred_probs = pred_outputs["lane_line_probs"][0].tolist()
    pred_lane_line_labels = [
        f"pred_lane_line_{i}@{prob:.2f}" for i, prob in enumerate(pred_probs)
    ]
    pred_lane_line_colors = [(0, int(255 * prob), 0) for prob in pred_probs]
    gt_lane_lines = to_3d(gt["lane_lines"][0])
    gt_probs = gt["lane_line_probs"][0].tolist()
    gt_lane_line_labels = [
        f"gt_lane_line_{i}@{prob:.2f}" for i, prob in enumerate(gt_probs)
    ]
    gt_lane_line_colors = [(int(255 * prob), 0, 0) for prob in gt_probs]

    create_plots(
        original_img,
        trajectories=pred_lane_lines + gt_lane_lines,
        labels=pred_lane_line_labels + gt_lane_line_labels,
        confs=pred_probs + gt_probs,
        colors=pred_lane_line_colors + gt_lane_line_colors,
        height=1.2,
        rpy=rpy_calib,
        save_path=(lane_lines_base_path / image_name).with_suffix(".png").as_posix(),
        title=f"Lane Lines, L2: {model_output_metrics['lane_lines_l2']:0.2f}",
    )

    # Plot road edges
    road_edges_base_path = base_path / "road_edges"
    road_edges_base_path.mkdir(parents=True, exist_ok=True)
    pred_road_edges = to_3d(pred_outputs["road_edges"][0])
    pred_road_edge_stds = stds_to_probs(pred_outputs["road_edge_stds"][0].tolist())
    pred_road_edge_labels = [
        f"pred_road_edge_{i}@{1 - std:.2f}" for i, std in enumerate(pred_road_edge_stds)
    ]
    pred_road_edge_colors = [(0, int(255 * prob), 0) for prob in pred_road_edge_stds]
    gt_road_edges = to_3d(gt["road_edges"][0])
    gt_road_edge_stds = stds_to_probs(gt["road_edge_stds"][0].tolist())
    gt_road_edge_labels = [
        f"gt_road_edge_{i}@{1 - std:.2f}" for i, std in enumerate(gt_road_edge_stds)
    ]
    gt_road_edge_colors = [(int(255 * prob), 0, 0) for prob in gt_road_edge_stds]
    create_plots(
        original_img,
        trajectories=pred_road_edges + gt_road_edges,
        labels=pred_road_edge_labels + gt_road_edge_labels,
        confs=pred_road_edge_stds + gt_road_edge_stds,
        colors=pred_road_edge_colors + gt_road_edge_colors,
        height=1.2,
        rpy=rpy_calib,
        save_path=(road_edges_base_path / image_name).with_suffix(".png").as_posix(),
        title=f"Road Edges, L2: {model_output_metrics['road_edges_l2']:0.2f}",
    )

    # Plot leading vehicles
    lead_base_path = base_path / "lead"
    lead_base_path.mkdir(parents=True, exist_ok=True)
    pred_leads = to_3d(pred_outputs["lead"][0][:, :, :2])
    pred_lead_probs = pred_outputs["lead_prob"][0].tolist()
    pred_lead_colors = [(0, int(255 * prob), 0) for prob in pred_lead_probs]
    pred_lead_labels = [
        f"pred_lead_{i}@{prob:.2f}" for i, prob in enumerate(pred_lead_probs)
    ]
    gt_leads = to_3d(gt["lead"][0][:, :, :2])
    gt_lead_probs = gt["lead_prob"][0].tolist()
    gt_lead_colors = [(int(255 * prob), 0, 0) for prob in gt_lead_probs]
    gt_lead_labels = [f"gt_lead_{i}@{prob:.2f}" for i, prob in enumerate(gt_lead_probs)]
    create_plots(
        original_img,
        trajectories=pred_leads + gt_leads,
        labels=pred_lead_labels + gt_lead_labels,
        confs=pred_lead_probs + gt_lead_probs,
        colors=pred_lead_colors + gt_lead_colors,
        height=1.2,
        rpy=rpy_calib,
        save_path=(lead_base_path / image_name).with_suffix(".png").as_posix(),
        title=f"Lead, L2: {model_output_metrics['lead_l2']:0.2f}",
        draw_on_img_func=draw_lead,
    )

    # Plot driving trajectory (position)
    pred_plan = pred_outputs["plan"]["position"].squeeze().cpu().numpy()
    plan_conf = float(pred_outputs["plan"]["position_prob"].item())
    gt_plan = gt["plan"]["position"].squeeze().cpu().numpy()
    plan_path = base_path / "plan"
    plan_path.mkdir(parents=True, exist_ok=True)
    plot_driving_trajectory(
        original_img,
        prev_input_img,
        current_input_img,
        pred_plan,
        gt_plan,
        plan_conf,
        rpy_calib,
        f"Plan, L2: {model_output_metrics['plan_l2']:0.2f}",
        (plan_path / image_name).with_suffix(".png").as_posix(),
    )
