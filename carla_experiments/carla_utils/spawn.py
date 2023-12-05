import random
from typing import Callable, Optional, Tuple, TypeVar, cast

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
    modify_blueprint_fn: Callable[
        [carla.ActorBlueprint], carla.ActorBlueprint
    ] = lambda x: x,
    on_measurement_received: Callable[[TSensorData], None] = lambda _: None,
) -> carla.Sensor:
    sensor_blueprint = world.get_blueprint_library().find(blueprint.name)
    sensor_blueprint = modify_blueprint_fn(sensor_blueprint)
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


def spawn_ego_vehicle(world: carla.World) -> carla.Vehicle:
    ego_bp = world.get_blueprint_library().find("vehicle.tesla.model3")
    ego_bp.set_attribute("role_name", "ego")
    # print("\nEgo role_name is set")
    ego_color = random.choice(ego_bp.get_attribute("color").recommended_values)
    ego_bp.set_attribute("color", ego_color)
    # print("\nEgo color is set")

    spawn_points = world.get_map().get_spawn_points()
    number_of_spawn_points = len(spawn_points)

    if 0 < number_of_spawn_points:
        random.shuffle(spawn_points)
        ego_transform = spawn_points[0]
        ego_vehicle = cast(carla.Vehicle, world.spawn_actor(ego_bp, ego_transform))
        print("\nEgo is spawned")
    else:
        raise Exception("Could not find any spawn points")

    # --------------
    # Spectator on ego position
    # --------------
    return ego_vehicle


def control_vehicle(
    vehicle: carla.Vehicle, throttle: float, steer: float, brake: float
) -> None:
    control = carla.VehicleControl(throttle, steer, brake)
    vehicle.apply_control(control)
