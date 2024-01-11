import random
import time
from typing import Callable, List, Optional, Tuple, TypeVar, cast

import carla

from carla_experiments.carla_utils.types_carla_utils import SensorBlueprint

TSensorData = TypeVar("TSensorData")


def spawn_sensor(
    world: carla.World,
    blueprint: SensorBlueprint[TSensorData],
    location: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    attach_to: Optional[carla.Actor] = None,
    attachment_type: carla.AttachmentType = carla.AttachmentType.Rigid,
    modify_blueprint_fn: Optional[
        Callable[[carla.ActorBlueprint], carla.ActorBlueprint]
    ] = None,
    on_measurement_received: Callable[[TSensorData], None] = lambda _: None,
) -> carla.Sensor:
    sensor_blueprint = world.get_blueprint_library().find(blueprint.name)
    sensor_blueprint = (
        modify_blueprint_fn(sensor_blueprint)
        if modify_blueprint_fn
        else sensor_blueprint
    )
    sensor_location = carla.Location(*location)
    sensor_rotation = carla.Rotation(*rotation)

    sensor_transform = carla.Transform(sensor_location, sensor_rotation)
    sensor_object = cast(
        carla.Sensor,
        world.spawn_actor(
            sensor_blueprint,
            sensor_transform,
            attach_to,
            attachment_type,
        ),
    )

    sensor_object.listen(on_measurement_received)

    return sensor_object


def spawn_ego_vehicle(
    world: carla.World,
    blueprint: str = "vehicle.tesla.model3",
    autopilot: bool = False,
    choose_spawn_point: Optional[
        Callable[[List[carla.Transform]], carla.Transform]
    ] = None,
) -> carla.Vehicle:
    ego_bp = world.get_blueprint_library().find(blueprint)
    ego_bp.set_attribute("role_name", "ego")
    ego_color = ego_bp.get_attribute("color").recommended_values[0]
    ego_bp.set_attribute("color", ego_color)

    spawn_points = world.get_map().get_spawn_points()
    number_of_spawn_points = len(spawn_points)

    if number_of_spawn_points > 0:
        ego_transform = (
            choose_spawn_point(spawn_points)
            if choose_spawn_point
            else random.choice(spawn_points)
        )
        ego_vehicle = cast(carla.Vehicle, world.spawn_actor(ego_bp, ego_transform))
    else:
        raise Exception("Could not find any spawn points")

    if autopilot:
        # Sleep before setting autopilot is important because of timing issues.
        time.sleep(1)
        world.tick()
        time.sleep(3)
        ego_vehicle.set_autopilot(True)
        print("Autopilot set")

    # --------------
    # Spectator on ego position
    # --------------
    return ego_vehicle


def control_vehicle(
    vehicle: carla.Vehicle, throttle: float, steer: float, brake: float
) -> None:
    control = carla.VehicleControl(throttle, steer, brake)
    vehicle.apply_control(control)
