from typing import TypedDict

import numpy as np

from carla_experiments.common.types_common import CarStatePartial
from carla_experiments.custom_logreader import log

DT_MDL = 0.05  # model


class Conversions:
    # Speed
    MPH_TO_KPH = 1.609344
    KPH_TO_MPH = 1.0 / MPH_TO_KPH
    MS_TO_KPH = 3.6
    KPH_TO_MS = 1.0 / MS_TO_KPH
    MS_TO_MPH = MS_TO_KPH * KPH_TO_MPH
    MPH_TO_MS = MPH_TO_KPH * KPH_TO_MS
    MS_TO_KNOTS = 1.9438
    KNOTS_TO_MS = 1.0 / MS_TO_KNOTS

    # Angle
    DEG_TO_RAD = np.pi / 180.0
    RAD_TO_DEG = 1.0 / DEG_TO_RAD

    # Mass
    LB_TO_KG = 0.453592


LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection

LANE_CHANGE_SPEED_MIN = 20 * Conversions.MPH_TO_MS
LANE_CHANGE_TIME_MAX = 10.0

DESIRES = {
    LaneChangeDirection.none: {
        LaneChangeState.off: log.Desire.none,
        LaneChangeState.preLaneChange: log.Desire.none,
        LaneChangeState.laneChangeStarting: log.Desire.none,
        LaneChangeState.laneChangeFinishing: log.Desire.none,
    },
    LaneChangeDirection.left: {
        LaneChangeState.off: log.Desire.none,
        LaneChangeState.preLaneChange: log.Desire.none,
        LaneChangeState.laneChangeStarting: log.Desire.laneChangeLeft,
        LaneChangeState.laneChangeFinishing: log.Desire.laneChangeLeft,
    },
    LaneChangeDirection.right: {
        LaneChangeState.off: log.Desire.none,
        LaneChangeState.preLaneChange: log.Desire.none,
        LaneChangeState.laneChangeStarting: log.Desire.laneChangeRight,
        LaneChangeState.laneChangeFinishing: log.Desire.laneChangeRight,
    },
}


class DesireHelper:
    def __init__(self):
        self.lane_change_state = LaneChangeState.off
        self.lane_change_direction = LaneChangeDirection.none
        self.lane_change_timer = 0.0
        self.lane_change_ll_prob = 1.0
        self.keep_pulse_timer = 0.0
        self.prev_one_blinker = False
        self.desire = log.Desire.none

    def update(
        self, carstate: CarStatePartial, lateral_active: bool, lane_change_prob: float
    ):
        v_ego = carstate["v_ego"]
        one_blinker = carstate["left_blinker"] != carstate["right_blinker"]
        below_lane_change_speed = v_ego < LANE_CHANGE_SPEED_MIN

        if not lateral_active or self.lane_change_timer > LANE_CHANGE_TIME_MAX:
            self.lane_change_state = LaneChangeState.off
            self.lane_change_direction = LaneChangeDirection.none
        else:
            # LaneChangeState.off
            if (
                self.lane_change_state == LaneChangeState.off
                and one_blinker
                and not self.prev_one_blinker
                and not below_lane_change_speed
            ):
                self.lane_change_state = LaneChangeState.preLaneChange
                self.lane_change_ll_prob = 1.0

            # LaneChangeState.preLaneChange
            elif self.lane_change_state == LaneChangeState.preLaneChange:
                # Set lane change direction
                self.lane_change_direction = (
                    LaneChangeDirection.left
                    if carstate["left_blinker"]
                    else LaneChangeDirection.right
                )

                torque_applied = carstate["steering_pressed"] and (
                    (
                        carstate["steering_torque"] > 0
                        and self.lane_change_direction == LaneChangeDirection.left
                    )
                    or (
                        carstate["steering_torque"] < 0
                        and self.lane_change_direction == LaneChangeDirection.right
                    )
                )

                blindspot_detected = (
                    carstate["left_blindspot"]
                    and self.lane_change_direction == LaneChangeDirection.left
                ) or (
                    carstate["right_blindspot"]
                    and self.lane_change_direction == LaneChangeDirection.right
                )

                if not one_blinker or below_lane_change_speed:
                    self.lane_change_state = LaneChangeState.off
                    self.lane_change_direction = LaneChangeDirection.none
                elif torque_applied and not blindspot_detected:
                    self.lane_change_state = LaneChangeState.laneChangeStarting

            # LaneChangeState.laneChangeStarting
            elif self.lane_change_state == LaneChangeState.laneChangeStarting:
                # fade out over .5s
                self.lane_change_ll_prob = max(
                    self.lane_change_ll_prob - 2 * DT_MDL, 0.0
                )

                # 98% certainty
                if lane_change_prob < 0.02 and self.lane_change_ll_prob < 0.01:
                    self.lane_change_state = LaneChangeState.laneChangeFinishing

            # LaneChangeState.laneChangeFinishing
            elif self.lane_change_state == LaneChangeState.laneChangeFinishing:
                # fade in laneline over 1s
                self.lane_change_ll_prob = min(self.lane_change_ll_prob + DT_MDL, 1.0)

                if self.lane_change_ll_prob > 0.99:
                    self.lane_change_direction = LaneChangeDirection.none
                    if one_blinker:
                        self.lane_change_state = LaneChangeState.preLaneChange
                    else:
                        self.lane_change_state = LaneChangeState.off

        if self.lane_change_state in (
            LaneChangeState.off,
            LaneChangeState.preLaneChange,
        ):
            self.lane_change_timer = 0.0
        else:
            self.lane_change_timer += DT_MDL

        self.prev_one_blinker = one_blinker

        self.desire = DESIRES[self.lane_change_direction][self.lane_change_state]

        # Send keep pulse once per second during LaneChangeStart.preLaneChange
        if self.lane_change_state in (
            LaneChangeState.off,
            LaneChangeState.laneChangeStarting,
        ):
            self.keep_pulse_timer = 0.0
        elif self.lane_change_state == LaneChangeState.preLaneChange:
            self.keep_pulse_timer += DT_MDL
            if self.keep_pulse_timer > 1.0:
                self.keep_pulse_timer = 0.0
            elif self.desire in (log.Desire.keepLeft, log.Desire.keepRight):
                self.desire = log.Desire.none


class DesireState(TypedDict):
    lane_change_state: int
    lane_change_direction: int
    lane_change_timer: float
    lane_change_ll_prob: float
    keep_pulse_timer: float
    prev_one_blinker: bool
    desire: int


INIT_DESIRE_STATE: DesireState = {
    "lane_change_state": LaneChangeState.off,
    "lane_change_direction": LaneChangeDirection.none,
    "lane_change_timer": 0.0,
    "lane_change_ll_prob": 1.0,
    "keep_pulse_timer": 0.0,
    "prev_one_blinker": False,
    "desire": log.Desire.none,
}


def get_next_desire_state(
    state: DesireState,
    carstate: CarStatePartial,
    lateral_active: bool,
    lane_change_prob: float,
) -> DesireState:
    new_state = state.copy()
    v_ego = carstate["v_ego"]
    one_blinker = carstate["left_blinker"] != carstate["right_blinker"]
    below_lane_change_speed = v_ego < LANE_CHANGE_SPEED_MIN

    if not lateral_active or state["lane_change_timer"] > LANE_CHANGE_TIME_MAX:
        new_state["lane_change_state"] = LaneChangeState.off
        new_state["lane_change_direction"] = LaneChangeDirection.none
    else:
        if (
            state["lane_change_state"] == LaneChangeState.off
            and one_blinker
            and not state["prev_one_blinker"]
            and not below_lane_change_speed
        ):
            new_state["lane_change_state"] = LaneChangeState.preLaneChange
            new_state["lane_change_ll_prob"] = 1.0

        elif state["lane_change_state"] == LaneChangeState.preLaneChange:
            new_state["lane_change_direction"] = (
                LaneChangeDirection.left
                if carstate["left_blinker"]
                else LaneChangeDirection.right
            )
            torque_applied = carstate["steering_pressed"] and (
                (
                    carstate["steering_torque"] > 0
                    and new_state["lane_change_direction"] == LaneChangeDirection.left
                )
                or (
                    carstate["steering_torque"] < 0
                    and new_state["lane_change_direction"] == LaneChangeDirection.right
                )
            )
            blindspot_detected = (
                carstate["left_blindspot"]
                and new_state["lane_change_direction"] == LaneChangeDirection.left
            ) or (
                carstate["right_blindspot"]
                and new_state["lane_change_direction"] == LaneChangeDirection.right
            )

            if not one_blinker or below_lane_change_speed:
                new_state["lane_change_state"] = LaneChangeState.off
                new_state["lane_change_direction"] = LaneChangeDirection.none
            elif torque_applied and not blindspot_detected:
                new_state["lane_change_state"] = LaneChangeState.laneChangeStarting

        elif state["lane_change_state"] == LaneChangeState.laneChangeStarting:
            new_state["lane_change_ll_prob"] = max(
                state["lane_change_ll_prob"] - 2 * DT_MDL, 0.0
            )
            if lane_change_prob < 0.02 and new_state["lane_change_ll_prob"] < 0.01:
                new_state["lane_change_state"] = LaneChangeState.laneChangeFinishing

        elif state["lane_change_state"] == LaneChangeState.laneChangeFinishing:
            new_state["lane_change_ll_prob"] = min(
                state["lane_change_ll_prob"] + DT_MDL, 1.0
            )
            if new_state["lane_change_ll_prob"] > 0.99:
                new_state["lane_change_direction"] = LaneChangeDirection.none
                new_state["lane_change_state"] = (
                    LaneChangeState.preLaneChange
                    if one_blinker
                    else LaneChangeState.off
                )

    if new_state["lane_change_state"] in (
        LaneChangeState.off,
        LaneChangeState.preLaneChange,
    ):
        new_state["lane_change_timer"] = 0.0
    else:
        new_state["lane_change_timer"] += DT_MDL

    new_state["prev_one_blinker"] = bool(one_blinker)
    new_state["desire"] = DESIRES[new_state["lane_change_direction"]][
        new_state["lane_change_state"]
    ]

    if new_state["lane_change_state"] in (
        LaneChangeState.off,
        LaneChangeState.laneChangeStarting,
    ):
        new_state["keep_pulse_timer"] = 0.0
    elif new_state["lane_change_state"] == LaneChangeState.preLaneChange:
        new_state["keep_pulse_timer"] += DT_MDL
        if new_state["keep_pulse_timer"] > 1.0:
            new_state["keep_pulse_timer"] = 0.0
        elif new_state["desire"] in (log.Desire.keepLeft, log.Desire.keepRight):
            new_state["desire"] = log.Desire.none

    return new_state
