from datetime import datetime
from functools import partial
from pathlib import Path
from typing import List, Literal, cast

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from openpilot_exploration.custom_logreader import log
from openpilot_exploration.datasets.openpilot_dataset import (
    OpenpilotDataset,
    get_desire_vector,
    get_dict_shape,
)
from openpilot_exploration.models.desire_helper import (
    INIT_DESIRE_STATE,
    get_next_desire_state,
)
from openpilot_exploration.models.supercombo_utils import (
    get_supercombo_onnx_model,
    mean_l2_loss,
    parse_supercombo_outputs,
    supercombo_tensors_at_idx,
    torch_dict_to_numpy,
)
from openpilot_exploration.openpilot_common.constants import SupercomboInputShapes
from openpilot_exploration.openpilot_common.types_common import (
    SupercomboEnv,
    SupercomboFullNumpyInputs,
    SupercomboOutputLogged,
    SupercomboPartialNumpyInput,
)
from openpilot_exploration.openpilot_common.utils_openpilot import (
    warp_image,
    yuv_6_channel_to_rgb,
)
from openpilot_exploration.openpilot_common.visualization import visualize_trajectory

PATH_TO_ONNX = Path(".weights/supercombo096.onnx")
PATH_TO_METADATA = Path(".weights/supercombo_metadata.pkl")  # not used


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


once = True


def transform_image(
    images: List[np.ndarray],
    senv: SupercomboEnv,
    image_type: Literal["narrow", "wide"],
) -> List[np.ndarray]:
    # TODO: Implement this correctly
    imgs = []
    for i, image in enumerate(images):
        env_indexed = supercombo_tensors_at_idx(senv, i, batched=False)
        # device_type = str(env_indexed["device_type"])
        # sensor = str(env_indexed["sensor"])
        # dc: DeviceCameraConfig = DEVICE_CAMERAS.get((device_type, sensor), None)  # type: ignore
        # if dc is None:
        #     raise ValueError(f"Unknown device_type: {device_type} and sensor: {sensor}")
        rpy_calib = env_indexed["rpy_calib"]
        # Is rpy calib really the extrinsic rotation
        # Additionally need translation matrix - speed?
        # extrinsic_matrix = euler2rot(rpy_calib.cpu().numpy())
        device_from_calib_euler = rpy_calib.cpu().numpy()

        if image_type == "narrow":
            some = warp_image(
                image,
                device_from_calib_euler,
            )
            imgs.append(some)
        elif image_type == "wide":
            some = warp_image(
                image,
                device_from_calib_euler,
                wide_camera=True,
                bigmodel_frame=True,
            )
            # print("some shape", some.shape)
            imgs.append(some)
        else:
            raise ValueError(f"Unknown image_type: {image_type}")
        # print("some\n", some)
        # plt.imsave("some.png", some)
        # sys.exit(0)
    # print("res shape", res.shape)

    # global once
    # image = images[0]
    # # print("image shape", image.shape)
    # if once:
    #     plt.imsave("image.png", image)
    #     once = False
    return imgs


def main_onnx():
    print("Using ONNX")
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
    base_save_path = Path(".plots") / current_date
    graphs_path = base_save_path / "graphs"
    base_save_path.mkdir(exist_ok=True, parents=True)
    base_save_path.mkdir(exist_ok=True, parents=True)
    graphs_path.mkdir(exist_ok=True, parents=True)

    model = SupercomboONNX()
    segment_start_idx = 300
    segment_end_idx = 450
    batch_size = 1  # CANNOT CHANGE THIS; HAS TO BE 1

    segment_length = segment_end_idx - segment_start_idx
    transform_narrow = partial(transform_image, image_type="narrow")
    transform_wide = partial(transform_image, image_type="wide")
    dataset = OpenpilotDataset(
        folder="/home/ulrikro/datasets/CommaAI/2024_03_27_Are",
        segment_start_idx=segment_start_idx,
        segment_end_idx=segment_end_idx,
        narrow_image_transforms=transform_narrow,
        wide_image_transforms=transform_wide,
        device="cpu",
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for i_batch, batch in tqdm(enumerate(dataloader), position=1):
        inputs_base, ground_truth, env = batch
        inputs_np = cast(
            SupercomboPartialNumpyInput,
            torch_dict_to_numpy(inputs_base, dtype=np.float32),
        )
        position_l2s = []
        lead_l2s = []
        lane_lines_l2s = []
        road_edges_l2s = []

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
        prev_desire = np.zeros(SupercomboInputShapes.DESIRES[1], dtype=np.float32)
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
            current_desire_vector = get_desire_vector(desire_state["desire"])
            # current_desire_vector = ground_truth["desire_state"].cpu().numpy()[0, 0]
            current_desire_vector[0] = 0
            # The current desire should not include previous desires,
            # and each element at each index should be either 0 or 1
            desire[:, -1] = np.where(
                current_desire_vector - prev_desire > 0.99, current_desire_vector, 0
            )
            # The main images are the wide images
            inputs: SupercomboFullNumpyInputs = {
                "big_input_imgs": partial_inputs["big_input_imgs"],
                "input_imgs": partial_inputs["input_imgs"],
                "traffic_convention": partial_inputs["traffic_convention"],
                "lateral_control_params": partial_inputs["lateral_control_params"],
                "desire": desire,
                "nav_features": np.zeros(
                    SupercomboInputShapes.NAV_FEATURES, dtype=np.float32
                ),
                "nav_instructions": np.zeros(
                    SupercomboInputShapes.NAV_INSTRUCTIONS, dtype=np.float32
                ),
                "prev_desired_curv": prev_desired_curv,
                "features_buffer": features_buffer,
            }
            gt: SupercomboOutputLogged = supercombo_tensors_at_idx(ground_truth, i)
            pred = model(inputs)
            parsed_pred = parse_supercombo_outputs(pred)
            features_buffer[:, :-1] = features_buffer[:, 1:]
            features_buffer[:, -1] = parsed_pred["hidden_state"]
            prev_desired_curv[:, :-1] = prev_desired_curv[:, 1:]
            prev_desired_curv[:, -1] = parsed_pred["desired_curvature"]
            prev_desire = current_desire_vector

            left_prob = parsed_pred["desire_state"][0, log.Desire.laneChangeLeft]
            right_prob = parsed_pred["desire_state"][0, log.Desire.laneChangeRight]
            # Updating lane_change_prob
            lane_change_prob = float(left_prob + right_prob)

            position_l2 = mean_l2_loss(
                parsed_pred["plan"]["position"], gt["plan"]["position"]
            )
            lead_l2 = mean_l2_loss(parsed_pred["lead"], gt["lead"])
            lane_lines_l2 = mean_l2_loss(parsed_pred["lane_lines"], gt["lane_lines"])
            road_edges_l2 = mean_l2_loss(parsed_pred["road_edges"], gt["road_edges"])
            position_l2s.append(position_l2)
            lead_l2s.append(lead_l2)
            lane_lines_l2s.append(lane_lines_l2)
            road_edges_l2s.append(road_edges_l2)

            # Plotting predicted trajectory
            trajectory_image = (
                partial_inputs["untransformed_narrow_imgs"].squeeze(0).astype(np.uint8)
            )
            rpy_calib = env_indexed["rpy_calib"].cpu().numpy().astype(np.float32)[0]
            # trajectory_image = warp_image(
            #     trajectory_image, rpy_calib, output_size=(512, 256)
            # )

            original_input_img = trajectory_image
            # yuv_6_channel_to_rgb(torch.tensor(trajectory_image).permute(0, 2, 3, 1))
            # .squeeze(0)
            # .to(dtype=torch.uint8)
            # .cpu()
            # .numpy()

            prev_input_img = (
                yuv_6_channel_to_rgb(
                    torch.tensor(partial_inputs["input_imgs"][:, :6]).permute(
                        0, 2, 3, 1
                    )
                )
                .squeeze(0)
                .to(dtype=torch.uint8)
                .cpu()
                .numpy()
            )
            current_input_img = (
                yuv_6_channel_to_rgb(
                    torch.tensor(partial_inputs["input_imgs"][:, 6:]).permute(
                        0, 2, 3, 1
                    )
                )
                .squeeze(0)
                .to(dtype=torch.uint8)
                .cpu()
                .numpy()
            )
            image_id = (i_batch + 1) * segment_length + i
            visualize_trajectory(
                original_input_img,
                prev_input_img,
                current_input_img,
                parsed_pred,
                gt,
                rpy_calib,
                base_save_path,
                image_name=str(image_id),
                model_output_metrics={
                    "plan_l2": position_l2,
                    "lead_l2": lead_l2,
                    "lane_lines_l2": lane_lines_l2,
                    "road_edges_l2": road_edges_l2,
                },
            )
        plt.plot(position_l2s, label="Position Mean L2")
        plt.savefig((graphs_path / f"batch_{i_batch}_position_l2.png").as_posix())
        plt.plot(lead_l2s, label="Lead L2")
        plt.savefig((graphs_path / f"batch_{i_batch}_lead_l2.png").as_posix())
        plt.plot(lane_lines_l2s, label="Lane Lines Mean L2")
        plt.savefig((graphs_path / f"batch_{i_batch}_lane_lines_l2.png").as_posix())
        plt.plot(road_edges_l2s, label="Road Edges Mean L2")
        plt.savefig((graphs_path / f"batch_{i_batch}_road_edges_l2.png").as_posix())


def main_plot():
    output = torch.load("output.pt")
    gt = torch.load("ground_truth.pt")
    inputs = torch.load("inputs.pt")
    both_img: np.ndarray = inputs["input_imgs"]
    print("inputs", get_dict_shape(inputs))
    prev_input_img = yuv_6_channel_to_rgb(
        torch.tensor(both_img[:, :6])
    )  # .squeeze(0).transpose(1, 2, 0)
    current_input_img = yuv_6_channel_to_rgb(
        torch.tensor(both_img[:, 6:])
    )  # .squeeze(0).transpose(1, 2, 0)

    # visualize_trajectory(
    #     current_input_img.cpu().numpy(),
    #     prev_input_img.cpu().numpy(),
    #     current_input_img.cpu().numpy(),
    #     output,
    #     gt,
    #     "pred",
    #     "pred_plan.png",
    # )
    # print("gt", gt)


if __name__ == "__main__":
    main_onnx()
