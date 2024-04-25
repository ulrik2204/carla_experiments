import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from carla_experiments.common.constants import SupercomboInputShapes
from carla_experiments.common.position_and_rotation import euler2rot
from carla_experiments.common.types_common import (
    SupercomboEnv,
    SupercomboFullNumpyInputs,
    SupercomboPartialNumpyInput,
    SupercomboPartialOutput,
)
from carla_experiments.common.utils_op_deepdive import calibrate_image
from carla_experiments.common.utils_openpilot import (
    DEVICE_CAMERAS,
    DeviceCameraConfig,
    yuv_6_channel_to_rgb,
)
from carla_experiments.common.visualization import visualize_trajectory
from carla_experiments.custom_logreader import log
from carla_experiments.datasets.comma3x_dataset import (
    Comma3xDataset,
    get_desire_vector,
    get_dict_shape,
)
from carla_experiments.models.desire_helper import (
    INIT_DESIRE_STATE,
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


def transform_image(
    images: torch.Tensor,
    senv: SupercomboEnv,
    image_type: Literal["narrow", "wide"],
) -> torch.Tensor:
    # TODO: Implement this correctly
    return images
    imgs = []
    print("images shapes before trans", images.shape)
    for i in range(images.shape[0]):
        image = images[i]
        # plt.imsave("before.png", image.cpu().to(dtype=torch.uint8).numpy())
        # sys.exit(0)
        print("senv", get_dict_shape(senv))
        env_indexed = supercombo_tensors_at_idx(senv, i, batched=False)
        device_type = str(env_indexed["device_type"])
        sensor = str(env_indexed["sensor"])
        dc: DeviceCameraConfig = DEVICE_CAMERAS.get((device_type, sensor), None)  # type: ignore
        if dc is None:
            raise ValueError(f"Unknown device_type: {device_type} and sensor: {sensor}")
        print("device_type", device_type)
        rpy_calib = env_indexed["rpy_calib"]
        extrinsic_matrix = euler2rot(rpy_calib.cpu().numpy())
        intrinsic_matrix = euler2rot(
            dc.fcam.intrinsics if image_type == "narrow" else dc.ecam.intrinsics
        )

        if image_type == "narrow":
            some = calibrate_image(
                image.to(dtype=torch.uint8).cpu().numpy(),
                extrinsic_matrix,
                intrinsic_matrix,
            )
            imgs.append(some)
        elif image_type == "wide":
            some = calibrate_image(
                image.to(dtype=torch.uint8).cpu().numpy(),
                rpy_calib.cpu().numpy(),
                intrinsic_matrix,
            )
            print("some shape", some.shape)
            imgs.append(some)
        else:
            raise ValueError(f"Unknown image_type: {image_type}")
        print("some\n", some)
        plt.imsave("some.png", some.to(dtype=torch.uint8).cpu().numpy())
        sys.exit(0)
    res = torch.stack(imgs)
    print("res shape", res.shape)
    return res


def main_onnx():
    print("Using ONNX")
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
    save_path = Path(".plots") / current_date
    save_path.mkdir(exist_ok=True, parents=True)
    model = SupercomboONNX()
    segment_start_idx = 300
    segment_end_idx = 400
    batch_size = 1  # CANNOT CHANGE THIS; HAS TO BE 1

    segment_length = segment_end_idx - segment_start_idx
    transform_narrow = partial(transform_image, image_type="narrow")
    transform_wide = partial(transform_image, image_type="wide")
    dataset = Comma3xDataset(
        folder="/home/ulrikro/datasets/CommaAI/2024_02_28_Orkdal",
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
            # print("loss", loss)

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
            print("prev_input_img", prev_input_img.shape)
            plt.imsave("prev_input_img.png", prev_input_img)
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
            visualize_trajectory(
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
    prev_input_img = yuv_6_channel_to_rgb(
        torch.tensor(both_img[:, :6])
    )  # .squeeze(0).transpose(1, 2, 0)
    current_input_img = yuv_6_channel_to_rgb(
        torch.tensor(both_img[:, 6:])
    )  # .squeeze(0).transpose(1, 2, 0)

    visualize_trajectory(
        prev_input_img.cpu().numpy(),
        current_input_img.cpu().numpy(),
        output,
        gt,
        0.5,
        "pred_plan.png",
    )
    # print("gt", gt)


if __name__ == "__main__":
    main_onnx()
