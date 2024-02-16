from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, TypeVar

import carla

from carla_experiments.carla_utils.spawn import spawn_ego_vehicle

TSensorData = TypeVar("TSensorData")


@dataclass
class CarlaContext:
    sensor_list: List[carla.Sensor]
    spectator: carla.Actor


def initialize_carla():
    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)
    client.load_world("Town01")
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True  # Enables synchronous mode
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    return client, world


def main():
    _, world = initialize_carla()
    spectator = world.get_spectator()
    carla_map = world.get_map()
    spawn_point_index = 64
    len_spawn_points = len(carla_map.get_spawn_points())
    used_spawn_point = carla_map.get_spawn_points()[spawn_point_index]
    ego_vehicle = spawn_ego_vehicle(world, spawn_point=used_spawn_point, autopilot=True)
    # world_snapshot =
    print(
        "Using spawn point",
        spawn_point_index,
        "of ",
        len_spawn_points,
        "with coords",
        f"({used_spawn_point.location.x}, {used_spawn_point.location.y}, {used_spawn_point.location.z})",
        "and rotation",
        f"({used_spawn_point.rotation.roll}, {used_spawn_point.rotation.pitch}, {used_spawn_point.rotation.yaw})",
    )
    # world.wait_for_tick()

    spectator = world.get_spectator()
    world.tick()
    print("mapname", world.get_map().name)

    while True:
        try:
            ego_trans = ego_vehicle.get_transform()
            used_trans = carla.Transform(
                ego_trans.location + carla.Location(z=2), ego_trans.rotation
            )
            spectator.set_transform(used_trans)

            world.tick()
        except KeyboardInterrupt:
            sys.exit()


if __name__ == "__main__":
    main()
