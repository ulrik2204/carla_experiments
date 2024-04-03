from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, Optional, TypedDict, Union

import cv2
import numpy as np
import torch
from capnp.lib.capnp import _DynamicStructReader
from PIL import Image
from torch.utils.data import Dataset

from carla_experiments.tools.lib.logreader import LogReader, ReadMode


class Comma3xModelInput(TypedDict):
    desire: torch.Tensor  # shape: [batch_size, 100, 8]
    traffic_convention: torch.Tensor  # shape: [batch_size, 2]
    lateral_control_params: torch.Tensor  # shape: [batch_size, 2]
    # prev_desired_curv: torch.Tensor  # shape: [batch_size, 100, 1], from model output
    nav_features: torch.Tensor  # shape: [batch_size, 256]
    nav_instructions: torch.Tensor  # shape: [batch_size, 150]
    # features_buffer: torch.Tensor  # shape: [batch_size, 99, 512], from model output
    input_imgs: torch.Tensor  # shape: [batch_size, 12, 128, 256]
    big_input_imgs: torch.Tensor  # shape: [batch_size, 12, 128, 256]


class Shapes:
    DESIRES = (100, 8)
    TRAFFIC_CONVENTION = (2,)
    LATERAL_CONTROL_PARAMS = (2,)
    PREV_DESIRED_CURV = (100, 1)
    NAV_FEATURES = (256,)
    NAV_INSTRUCTIONS = (150,)
    FEATURES_BUFFER = (99, 512)
    INPUT_IMGS = (12, 128, 256)
    BIG_INPUT_IMGS = (12, 128, 256)


@dataclass
class RlogImportantData:
    desireState: List[float]
    vEgo: float
    steerActuatorDelay: float
    navModelFeatures: List[float]
    navInstructionAllManeuvers: Optional[List[float]]
    # TODO: Continue on where to find the rest of the model input in OneNote before continuing here


def load_video(path: str) -> List[torch.Tensor]:
    """Loads a video from a file and returns it as a tensor.

    Args:
        path (str): The path to the video file.

    Returns:
        List[np.ndarray]: A list of frames in the video loaded in YUV format.
    """
    # TODO: use FrameReader instead
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # convert to PIL Image (converting from BGR to YUV)
        yuv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        frames.append(torch.tensor(yuv_image))
        # Process the frame
    cap.release()
    return frames


def load_log(path: str, mode: Literal["rlog", "qlog"]) -> List[_DynamicStructReader]:
    """Loads the log file and returns it as a list of strings.

    Args:
        path (str): The path to the log file.

    Returns:
        List[str]: A list of strings where each string is a line in the log file.
    """

    # TODO: HERE HERE: Load all LogReader relevant code into here
    read_mode = ReadMode.RLOG if mode == "rlog" else ReadMode.QLOG
    log = LogReader(path, default_mode=read_mode)
    return list(log)


def get_rlog_attr(rlog_list: list, attr: str, idx: int):
    # TODO: This is likely very slow, find a way to speed up
    relevant = [log for log in rlog_list if log.which() == attr]
    return relevant[idx]


def get_all_relevant_data_from_rlog(rlog_path: str) -> List[RlogImportantData]:
    rlog = load_log(rlog_path, mode="rlog")
    relevant_data: List[RlogImportantData] = []
    items = {
        "desireState": [],
        "vEgo": [],
        "steerActuatorDelay": [],
        "navModelFeatures": [],
        "allManeuvers": [],
    }
    for log in rlog:
        if log.which() == " modelV2":
            items["desireState"].append(log.modelV2.meta.desireState)
        if log.which() == "carState":
            items["vEgo"].append(log.carState.vEgo)
        if log.which() == "carParams":
            items["steerActuatorDelay"].append(log.carParams.steerActuatorDelay)
        if log.which() == "navModel":
            # This is only 2 Hz
            items["navModelFeatures"].append(log.navModel.features)
        if log.which() == "navInstruction":
            all_maneuvers = getattr(log.navInstruction, "allManeuvers", None)
            items["allManeuvers"].append(all_maneuvers)

    # Create the RlogImportantData objects
    for i, (desire, vEgo, delay) in enumerate(
        zip(
            items["desireState"],
            items["vEgo"],
            items["steerActuatorDelay"],
        )
    ):
        relevant_data.append(
            RlogImportantData(
                desireState=desire,
                vEgo=vEgo,
                steerActuatorDelay=delay,
                navModelFeatures=items["navModelFeatures"][i // 10],
                navInstructionAllManeuvers=items["allManeuvers"][i // 10],
            )
        )
    return relevant_data


class Comma3xDataset(Dataset):
    def __init__(
        self,
        folder: str,
        traffic_convention: Literal["right", "left"],
        device: str = "cuda",
    ) -> None:
        """Constructor for the Comma3xDataset class.


        Args:
            folder (str): The path to the folder containing each 1 min segment as folders.
        """
        self.device = device
        self.path = Path(folder)
        self.segment_paths = [item for item in self.path.iterdir() if item.is_dir()]
        # Assume the videos include the same amount of frames
        self.num_frames_per_video = len(
            load_video((self.segment_paths[0] / "ecamera.hevc").as_posix())
        )
        self.current_video_idx = 0
        self.current_wide_angle_frames = load_video(
            (self.segment_paths[self.current_video_idx] / "ecamera.hevc").as_posix()
        )
        self.current_narrow_frames = load_video(
            (self.segment_paths[self.current_video_idx] / "fcamera.hevc").as_posix()
        )

        # TODO: Do I need this?
        self.current_rlog_idx = 0
        self.current_rlog_relevant_data = get_all_relevant_data_from_rlog(
            (self.segment_paths[self.current_video_idx] / "rlog").as_posix()
        )

        # --- RLOG cached data ---
        self.traffic_convention = (
            torch.tensor([0.0, 1.0])
            if traffic_convention == "right"
            else torch.tensor([1.0, 0.0])
        )

        self.nav_features = torch.zeros(Shapes.NAV_FEATURES)
        self.nav_instructions = torch.zeros(Shapes.NAV_INSTRUCTIONS)

    def _get_video_frame_lazy(self, idx: int):
        segment_idx, frame_idx = np.divmod(idx, self.num_frames_per_video)
        segment_idx, frame_idx = int(segment_idx), int(frame_idx)
        if segment_idx == self.current_video_idx:
            return (
                self.current_wide_angle_frames[frame_idx],
                self.current_narrow_frames[frame_idx],
            )
        # Else: Load the new video into memory
        self.current_video_idx = segment_idx
        segment_path = self.segment_paths[segment_idx]
        ecamera_path = segment_path / "ecamera.hevc"
        fcamera_path = segment_path / "fcamera.hevc"
        # qcamera_path = device_path / "qcamera.ts" # not using
        # qlog_path = segment / "qlog" # only using rlog

        self.current_wide_angle_frames = load_video(ecamera_path.as_posix())
        self.current_narrow_frames = load_video(fcamera_path.as_posix())
        return (
            self.current_wide_angle_frames[frame_idx],
            self.current_narrow_frames[frame_idx],
        )

    def __len__(self) -> int:
        return self.num_frames_per_video * len(self.segment_paths)

    def _get_relevant_rlog_data_lazy(self, idx: int):
        if idx == self.current_rlog_idx:
            return self.current_rlog_relevant_data[idx]
        # Else: Load the new rlog into memory
        self.current_rlog_idx = idx
        self.current_rlog_relevant_data = get_all_relevant_data_from_rlog(
            (self.segment_paths[self.current_video_idx] / "rlog").as_posix()
        )
        return self.current_rlog_relevant_data[idx]

    def __getitem__(self, idx: int) -> Comma3xModelInput:
        # TODO: Currently not getting the last two frames, fix
        wide_angle_frame, narrow_frame = self._get_video_frame_lazy(idx)
        rlog_relevant = self._get_relevant_rlog_data_lazy(idx)
        current_and_previous_desires = torch.tensor(
            [
                item.desireState
                for item in reversed(self.current_rlog_relevant_data[: idx + 1])
            ]
        )
        # Desire
        # Pad desire with 0 vectors to reach 100
        desires = torch.zeros(Shapes.DESIRES)
        desires[: current_and_previous_desires.shape[0]] = current_and_previous_desires
        desires[:, 0] = 0.0
        # lateral_control_params
        v_ego = rlog_relevant.vEgo
        steer_actuator_delay = rlog_relevant.steerActuatorDelay
        # add 0.2 as they do in Openpilot for estimating other delays
        steer_delay = steer_actuator_delay + 0.2
        lateral_control_params = torch.tensor([v_ego, steer_delay])

        # TODO: Maybe change to tuple instead of dict depending on model?
        return {
            "desire": desires,
            "traffic_convention": self.traffic_convention,
            "lateral_control_params": lateral_control_params,
            # "prev_desired_curv": torch.zeros([100, 1]),  # TODO: Remove
            "nav_features": torch.tensor(rlog_relevant.navModelFeatures),
            "nav_instructions": torch.tensor(rlog_relevant.navInstructionAllManeuvers),
            # "features_buffer": torch.zeros([99, 512]),  # TODO: Remove
            # In Openpilot you can choose whether to mainly use narrow or wide frames, here maining narrow
            "input_imgs": narrow_frame.to(self.device),
            "big_input_imgs": wide_angle_frame.to(self.device),
        }


def main():
    dataset = Comma3xDataset(
        folder="/home/ulrikro/datasets/CommaAI/2024_02_28_Orkdal",
        traffic_convention="right",
        device="cuda",
    )
    print("Length of dataset:", len(dataset))
    item = dataset[0]
    print("First element\n", item)


if __name__ == "__main__":
    main()
