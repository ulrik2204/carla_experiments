from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import List, TypeVar

import carla

TSensorData = TypeVar("TSensorData")


@dataclass
class CarlaContext:
    sensor_list: List[carla.Sensor]
    spectator: carla.Actor


def initialize_carla():
    client = carla.Client("localhost", 2000)
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True  # Enables synchronous mode
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    return client, world


def main():
    _, world = initialize_carla()
    spectator = world.get_spectator()
    # world_snapshot =
    # world.wait_for_tick()

    spectator = world.get_spectator()
    count = 0

    while True:
        try:
            count += 1
            trans = spectator.get_transform()
            location = trans.location
            rot = trans.rotation
            time.sleep(0.01)
            if count % 10 == 0:
                print(
                    f"Location: {location.x=}, {location.y=}, {location.z=}, Rotation: {rot.yaw=}, {rot.pitch=}, {rot.roll=}"
                )

            world.tick()
        except KeyboardInterrupt:
            sys.exit()


if __name__ == "__main__":
    main()
