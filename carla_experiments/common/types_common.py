from typing import List, TypedDict

import numpy as np
import torch


class SupercomboPartialNumpyInput(TypedDict):
    traffic_convention: np.ndarray  # shape: [num_imgs, 2]
    lateral_control_params: np.ndarray  # shape: [num_imgs, 2]
    input_imgs: np.ndarray  # shape: [num_imgs, 12, 128, 256]
    big_input_imgs: np.ndarray  # shape: [num_imgs, 12, 128, 256]
    untransformed_narrow_imgs: np.ndarray  # shape: [num_imgs, 12, 128, 256]
    untransformed_wide_imgs: np.ndarray  # shape: [num_imgs, 12, 128, 256]


class SupercomboPartialTorchInput(TypedDict):
    traffic_convention: torch.Tensor  # shape: [num_imgs, 2]
    lateral_control_params: torch.Tensor  # shape: [num_imgs, 2]
    input_imgs: torch.Tensor  # shape: [num_imgs, 12, 128, 256]
    big_input_imgs: torch.Tensor  # shape: [num_imgs, 12, 128, 256]
    untransformed_narrow_imgs: torch.Tensor  # shape: [num_imgs, 12, 128, 256]
    untransformed_wide_imgs: torch.Tensor  # shape: [num_imgs, 12, 128, 256]


class SupercomboFullNumpyInputs(TypedDict):
    input_imgs: np.ndarray
    big_input_imgs: np.ndarray
    desire: np.ndarray
    traffic_convention: np.ndarray
    lateral_control_params: np.ndarray
    nav_features: np.ndarray
    nav_instructions: np.ndarray
    prev_desired_curv: np.ndarray
    features_buffer: np.ndarray


# Class for Torch Tensor inputs
class SupercomboFullTorchInputs(TypedDict):
    input_imgs: torch.Tensor
    big_input_imgs: torch.Tensor
    desire: torch.Tensor
    traffic_convention: torch.Tensor
    nav_features: torch.Tensor
    nav_instructions: torch.Tensor
    lateral_control_params: torch.Tensor
    prev_desired_curv: torch.Tensor
    features_buffer: torch.Tensor


class PlanTensors(TypedDict):
    position: torch.Tensor  # Shape([num_imgs, 33, 3])
    position_stds: torch.Tensor  # Shape([num_imgs, 33, 3])
    velocity: torch.Tensor  # Shape([num_imgs, 33, 3])
    acceleration: torch.Tensor  # Shape([num_imgs, 33, 3])
    t_from_current_euler: torch.Tensor  # Shape([num_imgs, 33, 3]), aka orientation
    orientation_rate: torch.Tensor  # Shape([num_imgs, 33, 3])


class PlanSliced(TypedDict):
    position: torch.Tensor  # Shape([num_imgs, 33, 3])
    velocity: torch.Tensor  # Shape([num_imgs, 33, 3])
    acceleration: torch.Tensor  # Shape([num_imgs, 33, 3])
    t_from_current_euler: torch.Tensor  # Shape([num_imgs, 33, 3]), aka orientation
    orientation_rate: torch.Tensor  # Shape([num_imgs, 33, 3])


class MetaSliced(TypedDict):
    engaged: torch.Tensor  # Shape([num_imgs, 1])
    brake_disengage: torch.Tensor  # Shape([num_imgs, 5])
    gas_disengage: torch.Tensor  # Shape([num_imgs, 5])
    steer_override: torch.Tensor  # Shape([num_imgs, 5])
    hard_brake_3: torch.Tensor  # Shape([num_imgs, 5])
    hard_brake_4: torch.Tensor  # Shape([num_imgs, 5])
    hard_brake_5: torch.Tensor  # Shape([num_imgs, 5])
    gas_press: torch.Tensor  # Shape([num_imgs, 5])
    left_blinker: torch.Tensor  # Shape([num_imgs, 6])
    right_blinker: torch.Tensor  # Shape([num_imgs, 6])


class MetaTensors(TypedDict):
    engaged_prob: torch.Tensor  # Shape([num_imgs, 1])
    brake_disengage_probs: torch.Tensor  # Shape([num_imgs, 5])
    gas_disengage_probs: torch.Tensor  # Shape([num_imgs, 5])
    steer_override_probs: torch.Tensor  # Shape([num_imgs, 5])
    brake_3_meters_per_second_squared_probs: torch.Tensor  # Shape([num_imgs, 5])
    brake_4_meters_per_second_squared_probs: torch.Tensor  # Shape([num_imgs, 5])
    brake_5_meters_per_second_squared_probs: torch.Tensor  # Shape([num_imgs, 5])
    # These are not accessible in RLOG
    # leftBlinkerProb: torch.Tensor  # Shape([num_imgs, 6])
    # rightBlinkerProb: torch.Tensor  # Shape([num_imgs, 6])


class PoseTensors(TypedDict):
    trans: torch.Tensor  # Shape([num_imgs, 3])
    rot: torch.Tensor  # Shape([num_imgs, 3])
    transStd: torch.Tensor  # Shape([num_imgs, 3])
    rotStd: torch.Tensor  # Shape([num_imgs, 3])


class SupercomboOutputLogged(TypedDict):
    plan: PlanTensors
    lane_lines: torch.Tensor  # Shape([num_imgs, 4, 33, 2])
    lane_line_probs: torch.Tensor  # Shape([num_imgs, 8])
    lane_line_stds: torch.Tensor  # Shape([num_imgs, 4])
    road_edges: torch.Tensor  # Shape([num_imgs, 2, 33, 2])
    road_edge_stds: torch.Tensor  # Shape([num_imgs, 2])
    lead: torch.Tensor  # Shape([num_imgs, 3, 6, 4])
    lead_stds: torch.Tensor  # Shape([num_imgs, 3, 6, 4])
    lead_prob: torch.Tensor  # Shape([num_imgs, 3])
    desire_state: torch.Tensor  # Shape([num_imgs, 8])
    meta: MetaTensors
    desire_pred: torch.Tensor  # Shape([num_imgs, 4, 8])
    pose: PoseTensors
    wide_from_device_euler: torch.Tensor  # Shape([num_imgs, 3]), only using the first 3
    wide_from_device_euler_std: torch.Tensor  # Shape([num_imgs, 3]) last 3
    sim_pose: PoseTensors
    road_transform: (
        torch.Tensor
    )  # Shape([num_imgs, 3]), comes from road_transofrm (of 6 first 3)
    road_transform_std: torch.Tensor  # Shape([num_imgs, 3]) last 3
    desired_curvature: torch.Tensor  # Shape([num_imgs, 1]), only using the first one


class PlanFull(TypedDict):
    position: torch.Tensor  # Shape([num_imgs, 33, 3])
    position_stds: torch.Tensor  # Shape([num_imgs, 33, 3])
    position_prob: torch.Tensor  # Shape([num_imgs, 1])
    velocity: torch.Tensor  # Shape([num_imgs, 33, 3])
    velocity_stds: torch.Tensor  # Shape([num_imgs, 33, 3])
    acceleration: torch.Tensor  # Shape([num_imgs, 33, 3])
    acceleration_stds: torch.Tensor  # Shape([num_imgs, 33, 3])
    t_from_current_euler: torch.Tensor  # Shape([num_imgs, 33, 3]), aka orientation
    t_from_current_euler_stds: torch.Tensor  # Shape([num_imgs, 33, 3])
    orientation_rate: torch.Tensor  # Shape([num_imgs, 33, 3])
    orientation_rate_stds: torch.Tensor  # Shape([num_imgs, 33, 3])


class SupercomboFullOutput(TypedDict):
    plan: PlanFull
    lane_lines: torch.Tensor  # Shape([num_imgs, 4, 33, 2])
    lane_line_probs: torch.Tensor  # Shape([num_imgs, 8])
    lane_line_stds: torch.Tensor  # Shape([num_imgs, 4])
    road_edges: torch.Tensor  # Shape([num_imgs, 2, 33, 2])
    road_edge_stds: torch.Tensor  # Shape([num_imgs, 2])
    lead: torch.Tensor  # Shape([num_imgs, 3, 6, 4])
    lead_stds: torch.Tensor  # Shape([num_imgs, 3, 6, 4])
    lead_prob: torch.Tensor  # Shape([num_imgs, 3])
    desire_state: torch.Tensor  # Shape([num_imgs, 8])
    meta: MetaTensors
    desire_pred: torch.Tensor  # Shape([num_imgs, 4, 8])
    pose: PoseTensors
    wide_from_device_euler: torch.Tensor  # Shape([num_imgs, 3]), only using the first 3
    wide_from_device_euler_std: torch.Tensor  # Shape([num_imgs, 3]) last 3
    sim_pose: PoseTensors
    road_transform: (
        torch.Tensor
    )  # Shape([num_imgs, 3]), comes from road_transofrm (of 6 first 3)
    road_transform_std: torch.Tensor  # Shape([num_imgs, 3]) last 3
    desired_curvature: torch.Tensor  # Shape([num_imgs, 1]), only using the first one
    hidden_state: torch.Tensor  # Shape([num_imgs, 512]


class FirstSliceSupercomboOutput(TypedDict):
    """Output of the model sliced. Original output size is [num_imgs, 6504]"""

    plan: torch.Tensor  # Shape([num_imgs, 33, 15])
    lane_lines: torch.Tensor  # Shape([num_imgs, 4, 33, 2])
    lane_line_probs: torch.Tensor  # Shape([num_imgs, 8])
    road_edges: torch.Tensor  # Shape([num_imgs, 2, 33, 2])
    lead: torch.Tensor  # Shape([num_imgs, 3, 6, 4])
    lead_prob: torch.Tensor  # Shape([num_imgs, 3])
    desire_state: torch.Tensor  # Shape([num_imgs, 8])
    meta: (
        torch.Tensor
    )  # Shape([num_imgs, 48]) (has a lot of subslices) # TODO: THIS IS UNPROCESSABLE
    desire_pred: torch.Tensor  # Shape([num_imgs, 4, 8])
    pose: torch.Tensor  # Shape([num_imgs, 12]) but only the first 6 are used
    wide_from_device_euler: (
        torch.Tensor
    )  # Shape([num_imgs, 6]), but only the first 3 are used
    sim_pose: torch.Tensor  # Shape([num_imgs, 12]), but only the first 6 are used
    road_transform: torch.Tensor  # Shape([num_imgs, 12]), but only the first 6 are used
    desired_curvature: (
        torch.Tensor
    )  # Shape([num_imgs, 2]), but only the first 1 is used
    hidden_state: torch.Tensor  # Shape([num_imgs, 512])


class CarStatePartial(TypedDict):
    v_ego: torch.Tensor  # Shape([num_imgs, 1]), type: float32
    steering_torque: torch.Tensor  # Shape([num_imgs, 1]), type: float32

    left_blinker: torch.Tensor  # Shape([num_imgs, 1]), type: bool
    right_blinker: torch.Tensor  # Shape([num_imgs, 1]), type: bool
    steering_pressed: torch.Tensor  # Shape([num_imgs, 1]), type: bool
    left_blindspot: torch.Tensor  # Shape([num_imgs, 1]), type: bool
    right_blindspot: torch.Tensor  # Shape([num_imgs, 1]), type: bool


class SupercomboEnv(TypedDict):
    car_state: CarStatePartial
    lateral_active: torch.Tensor  # Shape([num_imgs, 1]), type: bool
    rpy_calib: torch.Tensor  # Shape([num_imgs, 3]), type: float32
    device_type: List[str]  # Length num_imgs
    sensor: List[str]  # Length num_imgs


class SupercomboEnvIndexed(TypedDict):
    car_state: CarStatePartial
    lateral_active: torch.Tensor  # Shape([num_imgs, 1]), type: bool
    rpy_calib: torch.Tensor  # Shape([num_imgs, 3]), type: float32
    device_type: str  # Length num_imgs
    sensor: str  # Length num_imgs
