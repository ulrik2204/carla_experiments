from enum import Enum
from typing import Any, final


class _ConstantMeta(type):
    def __call__(cls, *args: Any, **kwargs: Any) -> None:
        raise TypeError(f"Cannot instantiate '{cls.__name__}' class")


class Constant(metaclass=_ConstantMeta):
    pass


class SupercomboInputShapes(Constant):
    DESIRES = (100, 8)
    TRAFFIC_CONVENTION = (2,)
    LATERAL_CONTROL_PARAMS = (2,)
    PREV_DESIRED_CURV = (100, 1)
    FEATURES_BUFFER = (99, 512)
    INPUT_IMGS = (12, 128, 256)
    BIG_INPUT_IMGS = (12, 128, 256)


SUPERCOMBO_OUTPUT_SLICES = {
    "plan": slice(0, 4955, None),
    "lane_lines": slice(4955, 5483, None),
    "lane_lines_prob": slice(5483, 5491, None),
    "road_edges": slice(5491, 5755, None),
    "lead": slice(5755, 5857, None),
    "lead_prob": slice(5857, 5860, None),
    "desire_state": slice(5860, 5868, None),
    "meta": slice(5868, 5916, None),
    "desire_pred": slice(5916, 5948, None),
    "pose": slice(5948, 5960, None),
    "wide_from_device_euler": slice(5960, 5966, None),
    "sim_pose": slice(5966, 5978, None),
    "road_transform": slice(5978, 5990, None),
    "desired_curvature": slice(5990, 5992, None),
    "hidden_state": slice(5992, None, None),
}

# Now you can access each slice using the dictionary keys, for example:


SUPERCOMBO_PLAN_SLICES = {
    "position": slice(0, 3),
    "velocity": slice(3, 6),
    "acceleration": slice(6, 9),
    "t_from_current_euler": slice(9, 12),
    "orientation_rate": slice(12, 15),
}

SUPERCOMBO_META_SLICES = {
    "engaged": slice(0, 1),
    "gas_disengage": slice(1, 36, 7),
    "brake_disengage": slice(2, 36, 7),
    "steer_override": slice(3, 36, 7),
    "hard_brake_3": slice(4, 36, 7),
    "hard_brake_4": slice(5, 36, 7),
    "hard_brake_5": slice(6, 36, 7),
    "gas_press": slice(7, 36, 7),
    "left_blinker": slice(36, 48, 2),
    "right_blinker": slice(37, 48, 2),
}


SUPERCOMBO_RESHAPES = {
    "plan": {"in_N": 5, "out_N": 1, "out_shape": (33, 15), "final_shape": (1, 33, 15)},
    "lane_lines": {
        "in_N": 0,
        "out_N": 0,
        "out_shape": (4, 33, 2),
        "final_shape": (1, 4, 33, 2),
    },
    "road_edges": {
        "in_N": 0,
        "out_N": 0,
        "out_shape": (2, 33, 2),
        "final_shape": (1, 2, 33, 2),
    },
    "lead": {"in_N": 2, "out_N": 3, "out_shape": (6, 4), "final_shape": (1, 3, 6, 4)},
    "pose": {"in_N": 0, "out_N": 0, "out_shape": (6,), "final_shape": (1, 6)},
    "sim_pose": {"in_N": 0, "out_N": 0, "out_shape": (6,), "final_shape": (1, 6)},
    "wide_from_device_euler": {
        "in_N": 0,
        "out_N": 0,
        "out_shape": (3,),
        "final_shape": (1, 3),
    },
    "road_transform": {"in_N": 0, "out_N": 0, "out_shape": (6,), "final_shape": (1, 6)},
    "desired_curvature": {
        "in_N": 0,
        "out_N": 0,
        "out_shape": (1,),
        "final_shape": (1, 1),
    },
    "hidden_state": None,  # no reshape
    "meta": None,  # binary crossentropy
    "lane_lines_prob": None,  # binary crossentropy
    "lead_prob": None,  # binary crossentropy
    "desire_state": None,  # categorical crossentropy
    "desire_pred": None,  # categorical crossentropy
}
