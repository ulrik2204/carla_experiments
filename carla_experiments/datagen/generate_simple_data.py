import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import TypedDict

import carla

from carla_experiments.carla_utils.constants import SensorBlueprints
from carla_experiments.carla_utils.setup import (
    game_loop_segment,
    setup_carla_client,
    setup_sensors,
)
from carla_experiments.carla_utils.spawn import spawn_ego_vehicle
from carla_experiments.carla_utils.types_carla_utils import BatchContext


class SimpleActorMap(TypedDict):
    ...


class SimpleSensorMap(TypedDict):
    camera: carla.Sensor


class SimpleSensorDataMap(TypedDict):
    camera: carla.Image


@dataclass
class SimpleContext(BatchContext[SimpleSensorMap, SimpleActorMap]):
    folder_base_path: Path
    images_base_path: Path
    controls_base_path: Path


start_time = time.time()


def check_stop_time_task(
    context: SimpleContext, sensor_data_map: SimpleSensorDataMap
) -> None:
    global start_time
    max_time_seconds = 60 * 60 * 3  # 3 hours
    if time.time() - start_time > max_time_seconds:
        print("Max time reached, exiting")
        raise KeyboardInterrupt()


def follow_camera_task(
    context: SimpleContext, sensor_data_map: SimpleSensorDataMap
) -> None:
    ego_vehicle = context.ego_vehicle
    vehicle_transform = ego_vehicle.get_transform()
    spectator = context.client.get_world().get_spectator()
    spectator_transform = spectator.get_transform()
    spectator_transform.location = vehicle_transform.location + carla.Location(z=4)
    spectator_transform.rotation.yaw = vehicle_transform.rotation.yaw
    spectator.set_transform(spectator_transform)


def save_data_task(
    context: SimpleContext, sensor_data_map: SimpleSensorDataMap
) -> None:
    ego_vehicle = context.ego_vehicle
    # world = context.client.get_world()
    control = ego_vehicle.get_control()
    # waypoint = world.get_map().get_waypoint(ego_vehicle.get_location())
    image = sensor_data_map["camera"]
    state = {
        "steer": control.steer,
        "throttle": control.throttle,
        "brake": control.brake,
        # "waypoint": waypoint
    }
    with open(context.controls_base_path / f"{image.frame:6d}.json", "w") as f:
        json.dump(state, f)
    image.save_to_disk((context.images_base_path / f"{image.frame:6d}.jpg").as_posix())


def main():
    client = setup_carla_client("Town10HD")  # type: ignore
    world = client.get_world()
    ego_vehicle = spawn_ego_vehicle(world, autopilot=True)
    sensor_data_queue = Queue()
    sensor_map = setup_sensors(
        world,
        ego_vehicle,
        sensor_data_queue,
        return_sensor_map_type=SimpleSensorMap,
        sensor_config={
            "camera": {
                "blueprint": SensorBlueprints.CAMERA_RGB,
                "location": (2, 0, 1),
                "rotation": (0, 0, 0),
                "attributes": {
                    "image_size_x": str(1920),
                    "image_size_y": str(1080),
                    "fov": str(105),
                },
            }
        },
    )
    # some: Tuple[str, carla.Waypoint] = client.get_trafficmanager().get_next_action(
    #     actors["ego_vehicle"]
    # )
    timestamp_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_base_path = Path(f"output/{timestamp_string}")
    folder_base_path.mkdir(parents=True, exist_ok=True)
    controls_base_path = folder_base_path / "controls"
    controls_base_path.mkdir(parents=True, exist_ok=True)
    images_base_path = folder_base_path / "images"
    images_base_path.mkdir(parents=True, exist_ok=True)

    context = SimpleContext(
        client=client,
        map=world.get_map(),
        sensor_map=sensor_map,
        sensor_data_queue=sensor_data_queue,
        actor_map={},
        ego_vehicle=ego_vehicle,
        folder_base_path=folder_base_path,
        images_base_path=images_base_path,
        controls_base_path=controls_base_path,
    )

    tasks = [follow_camera_task, save_data_task, check_stop_time_task]
    game_loop_segment(context, tasks)


if __name__ == "__main__":
    main()
