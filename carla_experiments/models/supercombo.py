from pathlib import Path
from typing import List, cast

import cv2
import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from carla_experiments.common.constants import SupercomboInputShapes
from carla_experiments.common.types_common import (
    CarStatePartial,
    SupercomboFullNumpyInputs,
    SupercomboPartialNumpyInput,
    SupercomboPartialOutput,
)
from carla_experiments.common.utils_op import img_from_device
from carla_experiments.custom_logreader import log
from carla_experiments.datasets.comma3x_dataset import (
    Comma3xDataset,
    RlogImportantData,
    get_desire_vector,
    get_dict_shape,
)
from carla_experiments.models.desire_helper import (
    INIT_DESIRE_STATE,
    DesireHelper,
    get_next_desire_state,
)
from carla_experiments.models.supercombo_utils import (
    get_supercombo_onnx_model,
    parse_supercombo_outputs,
    supercombo_tensors_at_idx,
    torch_dict_to_numpy,
    total_loss,
)

PATH_TO_ONNX = Path(".weights/supercombo.onnx")
PATH_TO_METADATA = Path(".weights/supercombo_metadata.pkl")


class SupercomboONNX:

    def __init__(self) -> None:

        # self.onnx_model = onnx.load(PATH_TO_ONNX.as_posix())
        # print("onnx_model", onnx.checker.check_model(self.onnx_model))
        self.sess = get_supercombo_onnx_model(PATH_TO_ONNX)
        # TODO: need to get CUDAExecutionProvider working

    def __repr__(self) -> str:
        return str(self.sess)

    def __call__(self, inputs: SupercomboFullNumpyInputs) -> np.ndarray:
        pred = self.sess.run(None, inputs)
        return pred[0]  # only use the first result which is the [1, 6504] tensor


def supercombo_image_to_rgb(input_array: np.ndarray) -> np.ndarray:
    """Reconstructs a YUV image from the processed array and plots it using cv2.

    Args:
        input_array (np.ndarray): A numpy array of shape (num_frames, height//2, width//2, 6)
            which has been processed as described.

    Returns:
        Plots the reconstructed YUV image using cv2.
    """
    (
        num_frames,
        channels,
        half_height,
        half_width,
    ) = input_array.shape
    full_height, full_width = 2 * half_height, 2 * half_width
    print("input shape", input_array.shape)

    # Create a numpy array for the reconstructed frames
    reconstructed_frames = np.zeros(
        (num_frames, 3, full_height, full_width), dtype=input_array.dtype
    )
    print("recon", reconstructed_frames.shape)

    for i in range(num_frames):
        # Create full resolution Y channel
        Y_full = np.zeros((full_height, full_width), dtype=input_array.dtype)
        Y_full[::2, ::2] = input_array[i, 0, :, :]  # Top-left
        Y_full[::2, 1::2] = input_array[i, 1, :, :]  # Top-right
        Y_full[1::2, ::2] = input_array[i, 2, :, :]  # Bottom-left
        Y_full[1::2, 1::2] = input_array[i, 3, :, :]  # Bottom-right

        # Upsample U and V channels using nearest neighbor
        U_full = np.repeat(np.repeat(input_array[i, 4, :, :], 2, axis=0), 2, axis=1)
        V_full = np.repeat(np.repeat(input_array[i, 5, :, :], 2, axis=0), 2, axis=1)

        # Assign channels back to reconstructed frames
        reconstructed_frames[i, 0, :, :] = Y_full
        reconstructed_frames[i, 1, :, :] = U_full
        reconstructed_frames[i, 2, :, :] = V_full

    # Convert the first frame to BGR for plotting since cv2 uses BGR format
    yuv_image = reconstructed_frames[0].astype(np.uint8).transpose(1, 2, 0)
    print("yuv_image", yuv_image.shape)
    rgb = np.array(cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB))
    return rgb


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
    print("conf", confs)
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
    img,
    width=1,
    height=1.2,
    fill_color=(128, 0, 255),
    line_color=(0, 255, 0),
):
    # device_path: N, 3
    device_path_l = device_path + np.array([0, 0, height])
    device_path_r = device_path + np.array([0, 0, height])
    device_path_l[:, 1] -= width
    device_path_r[:, 1] += width

    img_points_norm_l = img_from_device(device_path_l)
    img_points_norm_r = img_from_device(device_path_r)

    # img_pts_l = denormalize(img_points_norm_l)
    # img_pts_r = denormalize(img_points_norm_r)
    img_pts_l = img_points_norm_l
    img_pts_r = img_points_norm_r
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


def visualize_supercombo_outputs(
    prev_input_img: np.ndarray,
    current_input_img: np.ndarray,
    pred_outputs: SupercomboPartialOutput,
    gt: SupercomboPartialOutput,
    metric: float,
    save_file: str,
) -> None:
    pred_plan = pred_outputs["plan"]["position"].squeeze()
    print("pred plan size", pred_plan.shape)
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

    trajectories = [pred_plan] + [gt_plan]
    print("len trajectories", len(trajectories))
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


def main_onnx():
    print("Using ONNX")
    save_path = Path(".plots/")
    save_path.mkdir(exist_ok=True, parents=True)
    model = SupercomboONNX()
    segment_start_idx = 300
    segment_end_idx = 400
    batch_size = 1  # CANNOT CHANGE THIS; HAS TO BE 1

    segment_length = segment_end_idx - segment_start_idx
    dataset = Comma3xDataset(
        folder="/home/ulrikro/datasets/CommaAI/2024_02_28_Orkdal",
        segment_start_idx=segment_start_idx,
        segment_end_idx=segment_end_idx,
        device="cpu",
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for i_batch, batch in tqdm(enumerate(dataloader), position=1):
        inputs_base, ground_truth, env = batch
        inputs_np = cast(
            SupercomboPartialNumpyInput,
            torch_dict_to_numpy(inputs_base, dtype=np.float32),
        )
        # ground_truth_np = torch_dict_to_numpy(ground_truth, dtype=np.float32)

        # Recurrent inputs
        features_buffer = np.zeros(
            (batch_size,) + SupercomboInputShapes.FEATURES_BUFFER, dtype=np.float32
        )
        prev_desired_curv = np.zeros(
            (batch_size,) + SupercomboInputShapes.PREV_DESIRED_CURV, dtype=np.float32
        )
        desire = np.zeros(
            (batch_size,) + SupercomboInputShapes.DESIRES, dtype=np.float32
        )
        desire_state = INIT_DESIRE_STATE
        lane_change_prob = 0.0

        for i in tqdm(range(segment_length), position=2):
            partial_inputs = supercombo_tensors_at_idx(inputs_np, i)
            env_indexed = supercombo_tensors_at_idx(env, i)
            desire_state = get_next_desire_state(
                desire_state,
                env_indexed["car_state"],
                env_indexed["lateral_active"],
                lane_change_prob,
            )
            desire[:, :-1] = desire[:, 1:]
            desire[:, -1] = get_desire_vector(desire_state["desire"])
            inputs: SupercomboFullNumpyInputs = {
                "big_input_imgs": partial_inputs["big_input_imgs"],
                "input_imgs": partial_inputs["input_imgs"],
                "traffic_convention": partial_inputs["traffic_convention"],
                "lateral_control_params": partial_inputs["lateral_control_params"],
                "desire": desire,
                "prev_desired_curv": prev_desired_curv,
                "features_buffer": features_buffer,
            }
            gt = supercombo_tensors_at_idx(ground_truth, i)
            pred = model(inputs)
            parsed_pred = parse_supercombo_outputs(pred)
            features_buffer[:, :-1] = features_buffer[:, 1:]
            features_buffer[:, -1] = parsed_pred["hidden_state"]
            prev_desired_curv[:, :-1] = prev_desired_curv[:, 1:]
            prev_desired_curv[:, -1] = parsed_pred["desired_curvature"]

            left_prob = parsed_pred["desire_state"][0, log.Desire.laneChangeLeft]
            right_prob = parsed_pred["desire_state"][0, log.Desire.laneChangeRight]
            # Updating lane_change_prob
            lane_change_prob = float(left_prob + right_prob)

            # torch.save(inputs, "inputs.pt")
            # torch.save(parsed_pred, "output.pt")
            # torch.save(gt, "ground_truth.pt")
            compare_pred = cast(
                SupercomboPartialOutput,
                {
                    key: value
                    for key, value in parsed_pred.items()
                    if key != "hidden_state"
                },
            )
            loss = total_loss(compare_pred, gt)  # type: ignore
            print("loss", loss)

            # print("pred", pred)
            prev_input_img = supercombo_image_to_rgb(
                partial_inputs["input_imgs"][:, :6]
            )  # .squeeze(0).transpose(1, 2, 0)
            current_input_img = supercombo_image_to_rgb(
                partial_inputs["input_imgs"][:, 6:]
            )  # .squeeze(0).transpose(1, 2, 0)
            visualize_supercombo_outputs(
                prev_input_img,
                current_input_img,
                compare_pred,
                gt,
                loss,
                (save_path / f"pred_{i_batch}_{i}.png").as_posix(),
            )


def main_plot():
    output = torch.load("output.pt")
    gt = torch.load("ground_truth.pt")
    inputs = torch.load("inputs.pt")
    both_img: np.ndarray = inputs["input_imgs"]
    print("inputs", get_dict_shape(inputs))
    prev_input_img = supercombo_image_to_rgb(
        both_img[:, :6]
    )  # .squeeze(0).transpose(1, 2, 0)
    current_input_img = supercombo_image_to_rgb(
        both_img[:, 6:]
    )  # .squeeze(0).transpose(1, 2, 0)

    visualize_supercombo_outputs(
        prev_input_img, current_input_img, output, gt, 0.5, "pred_plan.png"
    )
    # print("gt", gt)


if __name__ == "__main__":
    main_onnx()
