import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import TypedDict

import carla
import numpy as np

from carla_experiments.carla_utils.constants import SensorBlueprints
from carla_experiments.carla_utils.measurements import (
    calculate_vehicle_speed,
    parse_gnss_data,
    parse_imu_data,
    parse_radar_data,
    parse_waypoint,
)
from carla_experiments.carla_utils.setup import (
    CarlaContext,
    game_loop,
    setup_carla_client,
    setup_sensors,
)
from carla_experiments.carla_utils.spawn import spawn_ego_vehicle


class AppActorsMap(TypedDict):
    ...


class AppSensorMap(TypedDict):
    front_camera: carla.Sensor
    radar: carla.Sensor
    gnss: carla.Sensor
    imu: carla.Sensor


class AppSensorDataMap(TypedDict):
    front_camera: carla.Image
    radar: carla.RadarMeasurement
    gnss: carla.GnssMeasurement
    imu: carla.IMUMeasurement


@dataclass
class AppContext(CarlaContext[AppSensorMap, AppActorsMap]):
    folder_base_path: Path
    images_base_path: Path
    radar_base_path: Path
    other_data_base_path: Path


def _save_dict_as_json(data: dict, path: Path):
    with path.open("w") as f:
        json.dump(data, f)


def save_data_task(context: AppContext, sensor_data_map: AppSensorDataMap) -> None:
    front_image = sensor_data_map["front_camera"]
    radar_data = parse_radar_data(sensor_data_map["radar"])
    imu_data = parse_imu_data(sensor_data_map["imu"])
    gnss_data = parse_gnss_data(sensor_data_map["gnss"])
    steering_angle = context.ego_vehicle.get_control().steer
    speed = calculate_vehicle_speed(context.ego_vehicle)
    frame = front_image.frame
    ego_vehicle = context.ego_vehicle
    waypoint = (
        context.client.get_world().get_map().get_waypoint(ego_vehicle.get_location())
    )
    other_data_dict = {
        "imu": imu_data,
        "gnss": gnss_data,
        "speed": speed,
        "steering_angle": steering_angle,
        "waypoint": parse_waypoint(waypoint),
    }
    front_image.save_to_disk(f"{context.images_base_path.as_posix()}/{frame:06d}.jpg")
    np.save(f"{context.radar_base_path.as_posix()}/{frame:06d}.npy", radar_data)
    _save_dict_as_json(
        other_data_dict, context.other_data_base_path / f"{frame:06d}.json"
    )


def spectator_follow_ego_vehicle_task(context: AppContext, _: AppSensorDataMap) -> None:
    ego_vehicle = context.ego_vehicle
    vehicle_transform = ego_vehicle.get_transform()
    spectator = context.client.get_world().get_spectator()
    spectator_transform = spectator.get_transform()
    spectator_transform.location = vehicle_transform.location + carla.Location(z=2)
    spectator_transform.rotation.yaw = vehicle_transform.rotation.yaw
    spectator.set_transform(spectator_transform)


def main():
    # save_folder,
    timestamp_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_base_path = Path(f"output/{timestamp_string}")
    folder_base_path.mkdir(parents=True, exist_ok=True)
    images_base_path = folder_base_path / "images"
    images_base_path.mkdir(parents=True, exist_ok=True)
    radar_base_path = folder_base_path / "radar"
    radar_base_path.mkdir(parents=True, exist_ok=True)
    other_data_base_path = folder_base_path / "other"
    other_data_base_path.mkdir(parents=True, exist_ok=True)

    client = setup_carla_client("Town04")
    world = client.get_world()
    ego_vehicle = spawn_ego_vehicle(
        world, autopilot=True, choose_spawn_point=lambda spawn_points: spawn_points[0]
    )
    sensor_data_queue = Queue()
    # TODO: Check sensor positions
    sensor_map = setup_sensors(
        world,
        ego_vehicle,
        sensor_data_queue=sensor_data_queue,
        sensor_config={
            "front_camera": {
                "blueprint": SensorBlueprints.CAMERA_RGB,
                "location": (2, 0, 1),
                "rotation": (0, 0, 0),
                "attributes": {},
            },
            "radar": {
                "blueprint": SensorBlueprints.RADAR_RANGE,
                "location": (2, 0, 1),
                "rotation": (0, 0, 0),
                "attributes": {},
            },
            "gnss": {
                "blueprint": SensorBlueprints.GNSS,
                "location": (2, 0, 1),
                "rotation": (0, 0, 0),
                "attributes": {},
            },
            "imu": {
                "blueprint": SensorBlueprints.IMU,
                "location": (2, 0, 1),
                "rotation": (0, 0, 0),
                "attributes": {},
            },
        },
    )

    context = AppContext(
        client=client,
        sensor_map=sensor_map,
        sensor_data_queue=sensor_data_queue,
        actor_map={},
        ego_vehicle=ego_vehicle,
        folder_base_path=folder_base_path,
        images_base_path=images_base_path,
        radar_base_path=radar_base_path,
        other_data_base_path=other_data_base_path,
    )
    print("App env: ", context)
    # TODO: How to handle config variables from the command line like save folder?
    # Should it be part of the CarlaSimulationEnvironment object?
    game_loop(context, [spectator_follow_ego_vehicle_task, save_data_task])


if __name__ == "__main__":
    main()
