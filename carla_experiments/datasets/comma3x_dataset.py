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
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from carla_experiments.common.types_common import (
    MetaTensors,
    PlanTensors,
    PoseTensors,
    SupercomboPartialOutput,
    SupercomboPartialTorchInput,
)
from carla_experiments.custom_logreader.custom_logreader import LogReader, ReadMode


@dataclass
class Position:
    x: List[float]  # length 33
    y: List[float]  # length 33
    z: List[float]  # length 33
    t: List[float]  # length 33
    xStd: List[float]  # length 33
    yStd: List[float]  # length 33
    zStd: List[float]  # length 33


@dataclass
class XYZT:
    x: List[float]  # length 33
    y: List[float]  # length 33
    z: List[float]  # length 33
    t: List[float]  # length 33


@dataclass
class DisengagePredictions:
    t: List[float]  # length 5
    brakeDisengageProbs: List[float]  # length 5
    gasDisengageProbs: List[float]  # length 5
    steerOverrideProbs: List[float]  # length 5
    brake3MetersPerSecondSquaredProbs: List[float]  # length 5
    brake4MetersPerSecondSquaredProbs: List[float]  # length 5
    brake5MetersPerSecondSquaredProbs: List[float]  # length 5


@dataclass
class MetaData:
    engagedProb: float
    desirePrediction: List[float]  # length 32
    desireState: List[float]  # length 8
    disengagePredictions: DisengagePredictions
    hardBrakePredicted: bool
    # laneChangeState: str
    # laneChangeDirection: str


@dataclass
class LeadsV3:
    prob: float
    probTime: float
    t: List[float]  # length 6
    x: List[float]  # length 6
    y: List[float]  # length 6
    v: List[float]  # length 6
    a: List[float]  # length 6
    xStd: List[float]  # length 6
    yStd: List[float]  # length 6
    vStd: List[float]  # length 6
    aStd: List[float]  # length 6


@dataclass
class TemporalPose:
    trans: List[float]  # length 3
    rot: List[float]  # length 3
    transStd: List[float]  # length 3
    rotStd: List[float]  # length 3


@dataclass
class Action:
    desiredCurvature: float


@dataclass
class ModelV2OutputData:
    position: Position
    orientation: XYZT
    velocity: XYZT
    orientationRate: XYZT
    laneLines: List[XYZT]  # length 4
    laneLineProbs: List[float]
    roadEdges: List[XYZT]  # length 2
    roadEdgeStds: List[float]  # length 2
    meta: MetaData
    laneLineStds: List[float]  # length 4
    # modelExecutionTime: float
    # gpuExecutionTime: float
    leadsV3: List[LeadsV3]  # length 3
    acceleration: XYZT
    # frameIdExtra: int
    temporalPose: TemporalPose
    # navEnabled: bool
    confidence: str
    # locationMonoTime: int
    action: Action


@dataclass
class CameraOdometryOutputData:
    trans: List[float]
    rot: List[float]
    transStd: List[float]
    rotStd: List[float]
    frameId: int
    timestampEof: int
    wideFromDeviceEuler: List[float]
    wideFromDeviceEulerStd: List[float]
    roadTransformTrans: List[float]
    roadTransformTransStd: List[float]


@dataclass
class RlogImportantData:
    vEgo: float
    steerActuatorDelay: float
    isRHD: bool
    modelV2: ModelV2OutputData
    cameraOdometry: CameraOdometryOutputData


def load_video(
    path: str, frame_shape: tuple = (512, 256), device: str = "cuda"
) -> List[torch.Tensor]:
    """Loads a video from a file and returns it as a tensor.

    Args:
        path (str): The path to the video file.
        frame_shape (tuple): The shape of the frames to reshape (width, height)

    Returns:
        List[np.ndarray]: A list of frames with size (height, width, 3) in YUV format.
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
        yuv_image = cv2.resize(yuv_image, frame_shape)
        tens = torch.tensor(yuv_image, device=device, dtype=torch.float32)
        frames.append(tens)
        # Process the frame
    cap.release()
    return frames


def preprocess_yuv_video_frames(frames: torch.Tensor) -> torch.Tensor:
    """Preprocess a tensor of video frames in YUV format.

    Args:
        frames (torch.Tensor): A tensor of shape (num_frames, height, width, 3)
            where the last dimension is the YUV channels.

    Returns:
        torch.Tensor: A tensor of shape (num_frames, height//2, width//2, 6)
    """
    num_frames, height, width, _ = frames.shape

    # Prepare output tensor
    output = torch.zeros(
        (num_frames, height // 2, width // 2, 6),
        dtype=frames.dtype,
        device=frames.device,
    )

    # Process each frame
    for i in range(num_frames):
        # Split Y, U, V channels
        Y = frames[i, :, :, 0]
        U = frames[i, :, :, 1]
        V = frames[i, :, :, 2]

        # Channel 0: Y[::2, ::2]
        output[i, :, :, 0] = Y[::2, ::2]
        # Channel 1: Y[::2, 1::2]
        output[i, :, :, 1] = Y[::2, 1::2]
        # Channel 2: Y[1::2, ::2]
        output[i, :, :, 2] = Y[1::2, ::2]
        # Channel 3: Y[1::2, 1::2]
        output[i, :, :, 3] = Y[1::2, 1::2]

        # Channel 4: U - at half resolution
        output[i, :, :, 4] = U[::2, ::2]
        # Channel 5: V - at half resolution
        output[i, :, :, 5] = V[::2, ::2]

    return output


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


def model_outputs_rlog_to_tensors(
    modelv2: ModelV2OutputData,
    camera_odometry: CameraOdometryOutputData,
    device: str = "cuda",
) -> SupercomboPartialOutput:
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
    position = torch.tensor(
        modelv2.position.x + modelv2.position.y + modelv2.position.z,
        device=device,
        dtype=torch.float32,
    ).reshape((len(modelv2.position.x), 3))
    position_stds = torch.tensor(
        modelv2.position.xStd + modelv2.position.yStd + modelv2.position.zStd,
        device=device,
        dtype=torch.float32,
    ).reshape((len(modelv2.position.x), 3))
    velocity = torch.tensor(
        modelv2.velocity.x + modelv2.velocity.y + modelv2.velocity.z,
        device=device,
        dtype=torch.float32,
    ).reshape((len(modelv2.velocity.x), 3))
    acceleration = torch.tensor(
        modelv2.acceleration.x + modelv2.acceleration.y + modelv2.acceleration.z,
        device=device,
        dtype=torch.float32,
    ).reshape((len(modelv2.acceleration.x), 3))
    t_from_current_euler = torch.tensor(
        [modelv2.orientation.x + modelv2.orientation.y + modelv2.orientation.z],
        device=device,
        dtype=torch.float32,
    ).reshape(len(modelv2.orientation.x), 3)
    orientation_rate = torch.tensor(
        modelv2.orientationRate.x
        + modelv2.orientationRate.y
        + modelv2.orientationRate.z,
        device=device,
        dtype=torch.float32,
    ).reshape((len(modelv2.orientationRate.x), 3))
    plan: PlanTensors = {
        "position": position,
        "position_stds": position_stds,
        "velocity": velocity,
        "acceleration": acceleration,
        "t_from_current_euler": t_from_current_euler,
        "orientation_rate": orientation_rate,
    }

    # Lane lines
    lane_lines = torch.tensor(
        [lane_line.x + lane_line.y for lane_line in modelv2.laneLines],
        device=device,
        dtype=torch.float32,
    ).reshape((len(modelv2.laneLines), 33, 2))
    lane_line_stds = torch.tensor(
        modelv2.laneLineStds, device=device, dtype=torch.float32
    )
    lane_line_probs = torch.tensor(
        modelv2.laneLineProbs, device=device, dtype=torch.float32
    )
    road_edges = torch.tensor(
        [road_edge.x + road_edge.y for road_edge in modelv2.roadEdges],
        device=device,
        dtype=torch.float32,
    ).reshape((len(modelv2.roadEdges), 33, 2))
    road_edge_stds = torch.tensor(
        modelv2.roadEdgeStds, device=device, dtype=torch.float32
    )
    lead = torch.tensor(
        [[lead.x, lead.y, lead.v, lead.a] for lead in modelv2.leadsV3],
        device=device,
        dtype=torch.float32,
    ).reshape((len(modelv2.leadsV3), 6, 4))
    lead_stds = torch.tensor(
        [[lead.xStd, lead.yStd, lead.vStd, lead.aStd] for lead in modelv2.leadsV3],
        device=device,
        dtype=torch.float32,
    ).reshape((len(modelv2.leadsV3), 6, 4))
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
        "vEgo": [],
        "steerActuatorDelay": [],
        "isRHD": [],
        "modelV2": [],
        "cameraOdometry": [],
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
            items["vEgo"].append(float(log.carState.vEgo))
        elif log.which() == "carParams":
            items["steerActuatorDelay"].append(float(log.carParams.steerActuatorDelay))
        elif log.which() == "driverMonitoringState":
            items["isRHD"].append(log.driverMonitoringState.isRHD)
    # Create the RlogImportantData objects
    relevant_data: List[RlogImportantData] = []
    for i in range(padding_before, num_frames - padding_after):
        modelv2 = get_item_by_frequency(items["modelV2"], i, num_frames, threshold)
        vEgo = get_item_by_frequency(items["vEgo"], i, num_frames, threshold)
        delay = get_item_by_frequency(
            items["steerActuatorDelay"], i, num_frames, threshold
        )
        isRHD = get_item_by_frequency(items["isRHD"], i, num_frames, threshold)
        cameraOdometry = get_item_by_frequency(
            items["cameraOdometry"], i, num_frames, threshold
        )
        relevant_data.append(
            RlogImportantData(
                vEgo=vEgo,
                steerActuatorDelay=delay,
                isRHD=isRHD,
                modelV2=modelv2,
                cameraOdometry=cameraOdometry,
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
        [log.vEgo for log in rlog_relevant], device=device, dtype=torch.float32
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


class Comma3xDataset(Dataset):
    def __init__(
        self,
        folder: str,
        segment_start_idx: int = 0,
        segment_end_idx: int = 1200,
        device: str = "cuda",
        image_transforms: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        """Constructor for the Comma3xDataset class.


        Args:
            folder (str): The path to the folder containing each 1 min segment as folders.
        """
        self.device = device
        self.path = Path(folder)
        self.segment_paths = [item for item in self.path.iterdir() if item.is_dir()]
        self.segment_start_idx = segment_start_idx
        self.segment_end_idx = segment_end_idx
        self.image_transforms = image_transforms
        # Assume the videos include the same amount of frames
        # self.num_frames_per_video = len(
        #     load_video((self.segment_paths[0] / "ecamera.hevc").as_posix())
        # )
        # self.current_frames: Optional[CurrentFrames] = None

        # self.current_rlog: Optional[CurrentRlog] = None

    def _get_video_frames(self, idx: int):
        # TODO: Preprocess images and cat the current image with the previous image
        segment_path = self.segment_paths[idx]
        ecamera_path = segment_path / "ecamera.hevc"
        fcamera_path = segment_path / "fcamera.hevc"
        # qcamera_path = device_path / "qcamera.ts" # not using
        # qlog_path = segment / "qlog" # only using rlog
        narrow_frames = preprocess_yuv_video_frames(
            torch.stack(
                load_video(fcamera_path.as_posix(), device=self.device)[
                    self.segment_start_idx : self.segment_end_idx
                ]
            )
        ).to(self.device)
        final_narrow_frames = (
            self.image_transforms(narrow_frames)
            if self.image_transforms is not None
            else narrow_frames
        )
        wide_angle_frames = preprocess_yuv_video_frames(
            torch.stack(
                load_video(ecamera_path.as_posix(), device=self.device)[
                    self.segment_start_idx : self.segment_end_idx
                ]
            )
        ).to(self.device)
        final_wide_angle_frames = (
            self.image_transforms(wide_angle_frames)
            if self.image_transforms is not None
            else wide_angle_frames
        )
        return (final_narrow_frames, final_wide_angle_frames)

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
    ) -> Tuple[SupercomboPartialTorchInput, SupercomboPartialOutput]:
        narrow_frames, wide_angle_frames = self._get_video_frames(idx)
        narrow_frames = concat_current_with_previous_frame(narrow_frames)
        wide_angle_frames = concat_current_with_previous_frame(wide_angle_frames)
        rlog_relevant = self._get_relevant_data_from_rlog(idx)
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
        }

        model_outputs_list = [
            model_outputs_rlog_to_tensors(
                log.modelV2, log.cameraOdometry, device=self.device
            )
            for log in rlog_relevant
        ]
        model_outputs_tensor_dict = default_collate(model_outputs_list)

        return (model_inputs, model_outputs_tensor_dict)


def get_dict_shape(d: Any):
    if type(d) is torch.Tensor or type(d) is np.ndarray:
        return d.shape
    return {key: get_dict_shape(value) for key, value in d.items()}


def main():
    print("Initializing the dataset")
    trans = transforms.Compose([transforms.Normalize(mean=[0.5], std=[0.5])])
    dataset = Comma3xDataset(
        folder="/home/ulrikro/datasets/CommaAI/2024_02_28_Orkdal",
        segment_start_idx=300,
        segment_end_idx=500,
        image_transforms=trans,
        device="cpu",
    )
    print("Printing dataset length")
    print("Length of dataset:", len(dataset))
    print("Getting first dataset element")
    model_input, gt = dataset[0]
    print("First input\n", get_dict_shape(model_input))
    print("First output\n", get_dict_shape(gt))


if __name__ == "__main__":
    main()
