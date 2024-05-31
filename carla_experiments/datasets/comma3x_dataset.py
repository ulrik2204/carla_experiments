from dataclasses import dataclass, fields
from pathlib import Path
from typing import (
    Any,
    Callable,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    TypedDict,
    TypeVar,
)

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from capnp.lib.capnp import _DynamicListReader, _DynamicStructReader
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from carla_experiments.common.constants import SupercomboInputShapes
from carla_experiments.common.rlog_types import (
    XYZT,
    CameraOdometryOutputData,
    CarState,
    DeviceState,
    LiveCalibration,
    ModelV2OutputData,
    Position,
    RoadCameraState,
)
from carla_experiments.common.types_common import (
    MetaTensors,
    PlanTensors,
    PoseTensors,
    SupercomboEnv,
    SupercomboEnvIndexed,
    SupercomboOutputLogged,
    SupercomboPartialTorchInput,
)
from carla_experiments.common.utils_openpilot import (
    rgb_to_6_channel_yuv,
    yuv_6_channel_to_rgb,
)
from carla_experiments.custom_logreader.custom_logreader import LogReader, ReadMode


@dataclass
class RlogImportantData:
    carState: CarState
    liveCalibration: LiveCalibration
    deviceState: DeviceState
    roadCameraState: RoadCameraState
    latActive: bool
    steerActuatorDelay: float
    isRHD: bool
    modelV2: ModelV2OutputData
    cameraOdometry: CameraOdometryOutputData


def load_video(
    path: str, frame_shape: tuple = (512, 256), device: str = "cuda"
) -> List[np.ndarray]:
    """Loads a video from a file and returns it as a tensor in RGB.

    Args:
        path (str): The path to the video file.
        frame_shape (tuple): The shape of the frames to reshape (width, height)

    Returns:
        List[torch.Tensor]: A list of uint8 tensor frames with size (height, width, 3)
            in RGB format.
    """
    # TODO: use FrameReader instead
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # convert to PIL Image (converting from BGR to YUV)
        rbg_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # yuv_image = cv2.resize(yuv_image, frame_shape)
        # tens = torch.tensor(yuv_image, device=device, dtype=torch.uint8)
        frames.append(np.array(rbg_image))
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

    # TODO: Load all LogReader relevant code into here
    # TODO: change to return videos instead?
    read_mode = ReadMode.RLOG if mode == "rlog" else ReadMode.QLOG
    log = LogReader(path, default_mode=read_mode)
    return list(log)


def get_rlog_attr(rlog_list: list, attr: str, idx: int):
    # TODO: This is likely very slow, find a way to speed up
    relevant = [log for log in rlog_list if log.which() == attr]
    return relevant[idx]


def to_xyzt(item: _DynamicStructReader):
    return XYZT(
        x=item.x,
        y=item.y,
        z=item.z,
        t=item.t,
    )


def to_position(item: _DynamicStructReader):
    return Position(
        x=item.x,
        y=item.y,
        z=item.z,
        t=item.t,
        xStd=item.xStd,
        yStd=item.yStd,
        zStd=item.zStd,
    )


T = TypeVar("T", bound=type)


def capnp_to_dataclass(item: _DynamicStructReader, cls: T) -> T:
    """Converts a capnp object to a dataclass.
    The _DynamicStructReader object must have all attributes
    as the dataclass cls requires including its nested types.
    """
    attrs = fields(cls)  # type: ignore
    inputs = {}
    for attr in attrs:
        name = attr.name
        at = getattr(item, name)
        if isinstance(at, _DynamicStructReader):
            at = capnp_to_dataclass(at, attr.type)
        if isinstance(at, _DynamicListReader):
            list_type = type(at[0])
            if list_type == _DynamicStructReader:
                at = [capnp_to_dataclass(item, attr.type.__args__[0]) for item in at]
            else:
                at = list(at)
        # if "list" in str(attr.type).lower():
        #     at = list(at)
        inputs[name] = at
    return cls(**inputs)


# def model_output_slicing(model_output: torch.Tensor) -> Comma3xModelOutput:
#     return


def stack_xyz(
    *args: List[float],
    device: str = "cuda",
    dtype=torch.float32,
) -> torch.Tensor:
    items = [torch.tensor(x, device=device, dtype=dtype) for x in args]
    return torch.stack(items, dim=1)


def model_outputs_rlog_to_tensors(
    modelv2: ModelV2OutputData,
    camera_odometry: CameraOdometryOutputData,
    device: str = "cuda",
) -> SupercomboOutputLogged:
    """Inputs a modelv2 object and returns the (1, 5992) size tensor
    including all the outputs from the model excluding the
    hidden state (which has size (1, 512)).


    Args:
        modelv2 (ModelV2OutputData): The modelv2 object to convert to tensor.

    Returns:
        torch.Tensor: The output tensor excluding the hidden state
    """
    # TODO: HERE HERE
    # Plan
    position = stack_xyz(
        modelv2.position.x,
        modelv2.position.y,
        modelv2.position.z,
        device=device,
        dtype=torch.float32,
    )
    position_stds = stack_xyz(
        modelv2.position.xStd,
        modelv2.position.yStd,
        modelv2.position.zStd,
        device=device,
        dtype=torch.float32,
    )
    velocity = stack_xyz(
        modelv2.velocity.x,
        modelv2.velocity.y,
        modelv2.velocity.z,
        device=device,
        dtype=torch.float32,
    )
    acceleration = stack_xyz(
        modelv2.acceleration.x,
        modelv2.acceleration.y,
        modelv2.acceleration.z,
        device=device,
        dtype=torch.float32,
    )
    t_from_current_euler = stack_xyz(
        modelv2.orientation.x,
        modelv2.orientation.y,
        modelv2.orientation.z,
        device=device,
        dtype=torch.float32,
    )
    orientation_rate = stack_xyz(
        modelv2.orientationRate.x,
        modelv2.orientationRate.y,
        modelv2.orientationRate.z,
        device=device,
        dtype=torch.float32,
    )
    plan: PlanTensors = {
        "position": position,
        "position_stds": position_stds,
        "velocity": velocity,
        "acceleration": acceleration,
        "t_from_current_euler": t_from_current_euler,
        "orientation_rate": orientation_rate,
    }

    # Lane lines
    lane_lines = torch.stack(
        [
            stack_xyz(
                lane_line.x,
                lane_line.y,
                device=device,
                dtype=torch.float32,
            )
            for lane_line in modelv2.laneLines
        ],
    )
    lane_line_stds = torch.tensor(
        modelv2.laneLineStds, device=device, dtype=torch.float32
    )
    lane_line_probs = torch.tensor(
        modelv2.laneLineProbs, device=device, dtype=torch.float32
    )
    road_edges = torch.stack(
        [
            stack_xyz(
                road_edge.x,
                road_edge.y,
                device=device,
                dtype=torch.float32,
            )
            for road_edge in modelv2.roadEdges
        ],
    )
    road_edge_stds = torch.tensor(
        modelv2.roadEdgeStds, device=device, dtype=torch.float32
    )
    lead = torch.stack(
        [
            stack_xyz(
                lead.x,
                lead.y,
                lead.v,
                lead.a,
                device=device,
                dtype=torch.float32,
            )
            for lead in modelv2.leadsV3
        ],
    )
    lead_stds = torch.stack(
        [
            stack_xyz(
                lead.xStd,
                lead.yStd,
                lead.vStd,
                lead.aStd,
                device=device,
                dtype=torch.float32,
            )
            for lead in modelv2.leadsV3
        ],
    )
    lead_prob = torch.tensor(
        [lead.prob for lead in modelv2.leadsV3], device=device, dtype=torch.float32
    )
    desire_state = torch.tensor(
        modelv2.meta.desireState, device=device, dtype=torch.float32
    )
    desire_pred = torch.tensor(
        modelv2.meta.desirePrediction, device=device, dtype=torch.float32
    ).reshape((4, 8))
    meta: MetaTensors = {
        "engaged_prob": torch.tensor(
            modelv2.meta.engagedProb, device=device, dtype=torch.float32
        ),
        "brake_disengage_probs": torch.tensor(
            modelv2.meta.disengagePredictions.brakeDisengageProbs,
            device=device,
            dtype=torch.float32,
        ),
        "gas_disengage_probs": torch.tensor(
            modelv2.meta.disengagePredictions.gasDisengageProbs,
            device=device,
            dtype=torch.float32,
        ),
        "steer_override_probs": torch.tensor(
            modelv2.meta.disengagePredictions.steerOverrideProbs,
            device=device,
            dtype=torch.float32,
        ),
        "brake_3_meters_per_second_squared_probs": torch.tensor(
            modelv2.meta.disengagePredictions.brake3MetersPerSecondSquaredProbs,
            device=device,
            dtype=torch.float32,
        ),
        "brake_4_meters_per_second_squared_probs": torch.tensor(
            modelv2.meta.disengagePredictions.brake4MetersPerSecondSquaredProbs,
            device=device,
            dtype=torch.float32,
        ),
        "brake_5_meters_per_second_squared_probs": torch.tensor(
            modelv2.meta.disengagePredictions.brake5MetersPerSecondSquaredProbs,
            device=device,
            dtype=torch.float32,
        ),
    }
    # TODO: NEED TO DEFINED camera_odometry first
    pose: PoseTensors = {
        "trans": torch.tensor(
            camera_odometry.trans, device=device, dtype=torch.float32
        ),
        "rot": torch.tensor(camera_odometry.rot, device=device, dtype=torch.float32),
        "transStd": torch.tensor(
            camera_odometry.transStd, device=device, dtype=torch.float32
        ),
        "rotStd": torch.tensor(
            camera_odometry.rotStd, device=device, dtype=torch.float32
        ),
    }
    wide_from_device_euler = torch.tensor(
        camera_odometry.wideFromDeviceEuler, device=device, dtype=torch.float32
    )
    wide_from_device_euler_std = torch.tensor(
        camera_odometry.wideFromDeviceEulerStd, device=device, dtype=torch.float32
    )
    sim_pose: PoseTensors = {
        "trans": torch.tensor(
            modelv2.temporalPose.trans, device=device, dtype=torch.float32
        ),
        "rot": torch.tensor(
            modelv2.temporalPose.rot, device=device, dtype=torch.float32
        ),
        "transStd": torch.tensor(
            modelv2.temporalPose.transStd, device=device, dtype=torch.float32
        ),
        "rotStd": torch.tensor(
            modelv2.temporalPose.rotStd, device=device, dtype=torch.float32
        ),
    }
    road_transform = torch.tensor(
        camera_odometry.roadTransformTrans, device=device, dtype=torch.float32
    )
    road_transform_std = torch.tensor(
        camera_odometry.roadTransformTransStd, device=device, dtype=torch.float32
    )
    desired_curvature = torch.tensor(
        modelv2.action.desiredCurvature, device=device, dtype=torch.float32
    )

    return {
        "plan": plan,
        "lane_lines": lane_lines,
        "lane_line_stds": lane_line_stds,
        "lane_line_probs": lane_line_probs,
        "road_edges": road_edges,
        "road_edge_stds": road_edge_stds,
        "lead": lead,
        "lead_stds": lead_stds,
        "lead_prob": lead_prob,
        "desire_state": desire_state,
        "desire_pred": desire_pred,
        "meta": meta,
        "pose": pose,
        "wide_from_device_euler": wide_from_device_euler,
        "wide_from_device_euler_std": wide_from_device_euler_std,
        "sim_pose": sim_pose,
        "road_transform": road_transform,
        "road_transform_std": road_transform_std,
        "desired_curvature": desired_curvature,
    }


def get_item_by_frequency(item, index, desired_length, threshold=10):
    item_length = len(item)
    if 0 <= abs(item_length - desired_length) < threshold:
        return item[index]
    return item[int(index * (item_length / desired_length))]


def get_all_relevant_data_from_rlog(
    rlog_path: str,
    padding_before: int = 100,
    padding_after: int = 100,
    num_frames: int = 1200,
) -> List[RlogImportantData]:
    threshold = (padding_before + padding_after) // 2
    rlog = load_log(rlog_path, mode="rlog")
    items = {
        "carState": [],
        "steerActuatorDelay": [],
        "isRHD": [],
        "modelV2": [],
        "cameraOdometry": [],
        "latActive": [],
        "liveCalibration": [],
        "deviceState": [],
        "roadCameraState": [],
    }
    for log in rlog:
        # The desireState input to the model is
        # the previous desireState output of the model
        # if log.which() == "modelV2":
        #     items["desireState"].append(log.modelV2.meta.desireState)
        if log.which() == "modelV2":
            data = capnp_to_dataclass(log.modelV2, ModelV2OutputData)
            items["modelV2"].append(data)
        elif log.which() == "cameraOdometry":
            data = capnp_to_dataclass(log.cameraOdometry, CameraOdometryOutputData)
            items["cameraOdometry"].append(data)
        elif log.which() == "carState":
            car_state = capnp_to_dataclass(log.carState, CarState)
            items["carState"].append(car_state)
        elif log.which() == "carParams":
            items["steerActuatorDelay"].append(float(log.carParams.steerActuatorDelay))
        elif log.which() == "carControl":
            items["latActive"].append(log.carControl.latActive)
        elif log.which() == "driverMonitoringState":
            items["isRHD"].append(log.driverMonitoringState.isRHD)
        elif log.which() == "liveCalibration":
            data = capnp_to_dataclass(log.liveCalibration, LiveCalibration)
            items["liveCalibration"].append(data)
        elif log.which() == "deviceState":
            data = capnp_to_dataclass(log.deviceState, DeviceState)
            items["deviceState"].append(data)
        elif log.which() == "roadCameraState":
            data = capnp_to_dataclass(log.roadCameraState, RoadCameraState)
            items["roadCameraState"].append(data)

    # Create the RlogImportantData objects
    relevant_data: List[RlogImportantData] = []
    for i in range(padding_before, num_frames - padding_after):
        modelv2 = get_item_by_frequency(items["modelV2"], i, num_frames, threshold)
        car_state = get_item_by_frequency(items["carState"], i, num_frames, threshold)
        delay = get_item_by_frequency(
            items["steerActuatorDelay"], i, num_frames, threshold
        )
        isRHD = get_item_by_frequency(items["isRHD"], i, num_frames, threshold)
        cameraOdometry = get_item_by_frequency(
            items["cameraOdometry"], i, num_frames, threshold
        )
        lat_active = get_item_by_frequency(items["latActive"], i, num_frames, threshold)
        live_calibration = get_item_by_frequency(
            items["liveCalibration"], i, num_frames, threshold
        )
        device_state = get_item_by_frequency(
            items["deviceState"], i, num_frames, threshold
        )
        road_camera_state = get_item_by_frequency(
            items["roadCameraState"], i, num_frames, threshold
        )
        relevant_data.append(
            RlogImportantData(
                carState=car_state,
                steerActuatorDelay=delay,
                isRHD=isRHD,
                modelV2=modelv2,
                cameraOdometry=cameraOdometry,
                latActive=lat_active,
                liveCalibration=live_calibration,
                deviceState=device_state,
                roadCameraState=road_camera_state,
            )
        )
    return relevant_data


def concat_current_with_previous_frame(frames_tensor: torch.Tensor):
    # Check if the input tensor has the correct shape

    # Initialize a list to hold the concatenated tensors
    concatenated_tensors = []

    # Concatenate the first image with itself
    concatenated_tensors.append(torch.cat([frames_tensor[0], frames_tensor[0]], dim=2))

    # Iterate over the remaining images
    for i in range(1, frames_tensor.size(0)):
        # Concatenate current image with the previous image along the channel axis
        concatenated_tensors.append(
            torch.cat([frames_tensor[i - 1], frames_tensor[i]], dim=2)
        )

    # Stack all concatenated images into a single tensor
    output_tensor = torch.stack(concatenated_tensors)

    return output_tensor


def get_lateral_control_params_tensor_from_rlog(
    rlog_relevant: List[RlogImportantData], device: str = "cuda"
):

    v_ego = torch.tensor(
        [log.carState.vEgo for log in rlog_relevant], device=device, dtype=torch.float32
    )
    steer_actuator_delay = torch.tensor(
        [log.steerActuatorDelay for log in rlog_relevant],
        device=device,
        dtype=torch.float32,
    )
    # add 0.2 as they do in Openpilot for estimating other delays
    steer_delay = steer_actuator_delay + 0.2
    lateral_control_params = torch.cat(
        (v_ego.unsqueeze(1), steer_delay.unsqueeze(1)), dim=1
    )
    return lateral_control_params


def get_traffic_conventions_tensor_from_rlog(
    rlog_relevant: List[RlogImportantData], device: str = "cuda"
):

    # traffic_convention
    return torch.stack(
        [
            torch.tensor(
                [0.0, 1.0] if log.isRHD else [1.0, 0.0],
                device=device,
                dtype=torch.float32,
            )
            for log in rlog_relevant
        ],
    )


def get_desire_vector(desire: int) -> np.ndarray:
    vec_desire = np.zeros(SupercomboInputShapes.DESIRES[1], dtype=np.float32)
    if desire >= 0 and desire < SupercomboInputShapes.DESIRES[1]:
        vec_desire[desire] = 1
    return vec_desire


class Comma3xDataset(Dataset):
    def __init__(
        self,
        folder: str,
        segment_start_idx: int = 0,
        segment_end_idx: int = 1200,
        device: str = "cuda",
        narrow_image_transforms: Optional[
            Callable[[List[np.ndarray], SupercomboEnv], List[np.ndarray]]
        ] = None,
        wide_image_transforms: Optional[
            Callable[[List[np.ndarray], SupercomboEnv], List[np.ndarray]]
        ] = None,
    ) -> None:
        """Constructor for the Comma3xDataset class.

        Args:
            folder (str): The folder to the folder of comma3x segments.
                Each segment should contain at least the files: ecamera.hevc, fcamera.hevc, rlog
            segment_start_idx (int, optional): The camera frame index each segment returned should start at.
                Defaults to 0.
            segment_end_idx (int, optional): The camera frame index each segment returned should end at.
                Defaults to 1200.
            device (str, optional): torch device. Defaults to "cuda".
            narrow_image_transforms (Optional[Callable[[List[np.ndarray]], List[np.ndarray]] ], optional):
                Transforms taking in a list of images (original dimensions) as RGB image and returns
                the list of transformed RGB images. These will later be transformed to (h:256, w:512).
                Defaults to None.
            wide_image_transforms (Optional[Callable[[List[np.ndarray]], List[np.ndarray]]], optional):
                Transforms taking in a list of images (original dimensions) as RGB image and returns
                the list of transformed RGB images. These will later be transformed to (h:256, w:512).
                Defaults to None.
        """
        self.device = device
        self.path = Path(folder)
        self.segment_paths = [item for item in self.path.iterdir() if item.is_dir()]
        self.segment_start_idx = segment_start_idx
        self.segment_end_idx = segment_end_idx
        self.narrow_image_transforms = narrow_image_transforms
        self.wide_image_transforms = wide_image_transforms
        # Assume the videos include the same amount of frames
        # self.num_frames_per_video = len(
        #     load_video((self.segment_paths[0] / "ecamera.hevc").as_posix())
        # )
        # self.current_frames: Optional[CurrentFrames] = None

        # self.current_rlog: Optional[CurrentRlog] = None

    def _get_video_frames(self, idx: int, senv: SupercomboEnv):
        # TODO: Preprocess images and cat the current image with the previous image
        segment_path = self.segment_paths[idx]
        ecamera_path = segment_path / "ecamera.hevc"
        fcamera_path = segment_path / "fcamera.hevc"
        # qcamera_path = device_path / "qcamera.ts" # not using
        # qlog_path = segment / "qlog" # only using rlog
        narrow_frames_rgb = load_video(fcamera_path.as_posix(), device=self.device)[
            self.segment_start_idx : self.segment_end_idx
        ]
        transformed_narrow_frames = (
            self.narrow_image_transforms(narrow_frames_rgb, senv)
            if self.narrow_image_transforms is not None
            else narrow_frames_rgb
        )
        wide_angle_frames = load_video(ecamera_path.as_posix(), device=self.device)[
            self.segment_start_idx : self.segment_end_idx
        ]
        transformed_wide_angle_frames = (
            self.wide_image_transforms(wide_angle_frames, senv)
            if self.wide_image_transforms is not None
            else wide_angle_frames
        )
        resized_original_narrow = [
            cv2.resize(frame, (512, 256)) for frame in narrow_frames_rgb
        ]
        resized_narrow = [
            cv2.resize(frame, (512, 256)) for frame in transformed_narrow_frames
        ]
        resized_wide = [
            cv2.resize(frame, (512, 256)) for frame in transformed_wide_angle_frames
        ]
        stacked_original_narrow = torch.stack(
            [
                torch.tensor(frame, device=self.device, dtype=torch.uint8)
                for frame in resized_original_narrow
            ]
        )
        stacked_narrow = torch.stack(
            [
                torch.tensor(frame, device=self.device, dtype=torch.uint8)
                for frame in resized_narrow
            ]
        )
        stacked_wide = torch.stack(
            [
                torch.tensor(frame, device=self.device, dtype=torch.uint8)
                for frame in resized_wide
            ]
        )
        final_narrow_frames = rgb_to_6_channel_yuv(stacked_narrow)
        final_wide_angle_frames = rgb_to_6_channel_yuv(stacked_wide)
        final_original_narrow = rgb_to_6_channel_yuv(stacked_original_narrow)
        return (
            final_narrow_frames.to(dtype=torch.float32),
            final_wide_angle_frames.to(dtype=torch.float32),
            final_original_narrow,
        )

    def __len__(self) -> int:
        return len(self.segment_paths)

    def _get_relevant_data_from_rlog(self, idx: int):
        return get_all_relevant_data_from_rlog(
            (self.segment_paths[idx] / "rlog").as_posix(),
            num_frames=1200,
            padding_before=100,
            padding_after=100,
        )[self.segment_start_idx : self.segment_end_idx]

    def __getitem__(
        self, idx: int
    ) -> Tuple[SupercomboPartialTorchInput, SupercomboOutputLogged, SupercomboEnv]:
        rlog_relevant = self._get_relevant_data_from_rlog(idx)
        supercombo_env: SupercomboEnv = {
            "lateral_active": torch.tensor(
                [log.latActive for log in rlog_relevant],
                device=self.device,
                dtype=torch.float32,
            ),
            "car_state": {
                "v_ego": torch.tensor(
                    [log.carState.vEgo for log in rlog_relevant],
                    device=self.device,
                    dtype=torch.bool,
                ),
                "steering_torque": torch.tensor(
                    [log.carState.steeringTorque for log in rlog_relevant],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "left_blindspot": torch.tensor(
                    [log.carState.leftBlindspot for log in rlog_relevant],
                    device=self.device,
                    dtype=torch.bool,
                ),
                "right_blindspot": torch.tensor(
                    [log.carState.rightBlindspot for log in rlog_relevant],
                    device=self.device,
                    dtype=torch.bool,
                ),
                "left_blinker": torch.tensor(
                    [log.carState.leftBlinker for log in rlog_relevant],
                    device=self.device,
                    dtype=torch.bool,
                ),
                "right_blinker": torch.tensor(
                    [log.carState.rightBlinker for log in rlog_relevant],
                    device=self.device,
                    dtype=torch.bool,
                ),
                "steering_pressed": torch.tensor(
                    [log.carState.steeringPressed for log in rlog_relevant],
                    device=self.device,
                    dtype=torch.bool,
                ),
            },
            "rpy_calib": torch.tensor(
                [log.liveCalibration.rpyCalib for log in rlog_relevant],
                device=self.device,
                dtype=torch.float32,
            ),
            "device_type": [str(log.deviceState.deviceType) for log in rlog_relevant],
            "sensor": [str(log.roadCameraState.sensor) for log in rlog_relevant],
        }
        narrow_frames, wide_angle_frames, original_narrows = self._get_video_frames(
            idx, supercombo_env
        )
        narrow_frames = concat_current_with_previous_frame(narrow_frames)
        wide_angle_frames = concat_current_with_previous_frame(wide_angle_frames)
        original_narrows = concat_current_with_previous_frame(original_narrows)
        traffic_convention = get_traffic_conventions_tensor_from_rlog(
            rlog_relevant, device=self.device
        )
        lateral_control_params = get_lateral_control_params_tensor_from_rlog(
            rlog_relevant, device=self.device
        )

        # TODO: Maybe change to tuple instead of dict depending on model?
        model_inputs: SupercomboPartialTorchInput = {
            # "desire": desires,
            "traffic_convention": traffic_convention,
            "lateral_control_params": lateral_control_params,
            # "prev_desired_curv": torch.zeros([100, 1]),  # TODO: Remove
            # "features_buffer": torch.zeros([99, 512]),  # TODO: Remove
            # In Openpilot you can choose whether to mainly use narrow or wide frames, here maining narrow
            "input_imgs": narrow_frames.permute(0, 3, 1, 2),
            "big_input_imgs": wide_angle_frames.permute(0, 3, 1, 2),
            "untransformed_narrow_imgs": original_narrows.permute(0, 3, 1, 2),
        }

        model_outputs_list = [
            model_outputs_rlog_to_tensors(
                log.modelV2, log.cameraOdometry, device=self.device
            )
            for log in rlog_relevant
        ]
        model_outputs_tensor_dict = default_collate(model_outputs_list)

        return (model_inputs, model_outputs_tensor_dict, supercombo_env)


def get_dict_shape(d: Any):
    if type(d) is torch.Tensor or type(d) is np.ndarray:
        return d.shape
    if type(d) is list:
        return "List of length " + str(len(d))
    return {key: get_dict_shape(value) for key, value in d.items()}


def main():
    print("Initializing the dataset")
    dataset = Comma3xDataset(
        folder="/home/ulrikro/datasets/CommaAI/2024_02_28_Orkdal",
        segment_start_idx=300,
        segment_end_idx=500,
        device="cpu",
    )
    print("Printing dataset length")
    print("Length of dataset:", len(dataset))
    print("Getting first dataset element")
    model_input, gt, _ = dataset[0]
    print("First input\n", get_dict_shape(model_input))
    print("First output\n", get_dict_shape(gt))


if __name__ == "__main__":
    main()
