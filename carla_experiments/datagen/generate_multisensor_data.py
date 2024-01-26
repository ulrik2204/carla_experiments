from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import TypedDict

import carla

from carla_experiments.carla_utils.constants import SensorBlueprints
from carla_experiments.carla_utils.setup import (
    BatchContext,
    game_loop_segment,
    setup_carla_client,
    setup_sensors,
)
from carla_experiments.carla_utils.spawn import spawn_ego_vehicle


class AppActorsMap(TypedDict):
    ...


class AppSensorMap(TypedDict):
    front_camera: carla.Sensor
    back_camera: carla.Sensor
    lidar: carla.Sensor


class AppSensorDataMap(TypedDict):
    front_camera: carla.Image
    back_camera: carla.Image
    lidar: carla.LidarMeasurement


@dataclass  # kw_only=True
class AppContext(BatchContext[AppSensorMap, AppActorsMap]):
    folder_base_path: Path
    images_base_path: Path
    controls_base_path: Path


def save_image_task(context: AppContext, sensor_data_map: AppSensorDataMap) -> None:
    front_image = sensor_data_map["front_camera"]
    back_image = sensor_data_map["back_camera"]
    lidar_point_cloud = sensor_data_map["lidar"]
    print("Front image, frame: ", front_image, front_image.frame)
    print("Back image, frame: ", back_image, back_image.frame)
    print("Lidar point cloud, frame: ", lidar_point_cloud, lidar_point_cloud.frame)
    front_image.save_to_disk(
        f"{context.images_base_path.as_posix()}/font_{front_image.frame:06d}.jpg"
    )
    back_image.save_to_disk(
        f"{context.images_base_path.as_posix()}/back_{back_image.frame:06d}.jpg"
    )
    lidar_point_cloud.save_to_disk(
        f"{context.images_base_path.as_posix()}/lidar_{back_image.frame:06d}.ply"
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
    controls_base_path = folder_base_path / "controls"
    controls_base_path.mkdir(parents=True, exist_ok=True)
    images_base_path = folder_base_path / "images"
    images_base_path.mkdir(parents=True, exist_ok=True)

    client = setup_carla_client("Town04")
    world = client.get_world()
    carla_map = world.get_map()
    ego_vehicle = spawn_ego_vehicle(
        world, autopilot=True, spawn_point=carla_map.get_spawn_points()[0]
    )
    sensor_data_queue = Queue()
    sensor_map = setup_sensors(
        world,
        ego_vehicle,
        sensor_data_queue=sensor_data_queue,
        return_sensor_map_type=AppSensorMap,
        sensor_config={
            "front_camera": {
                "blueprint": SensorBlueprints.CAMERA_RGB,
                "location": (2, 0, 1),
                "rotation": (0, 0, 0),
                "attributes": {},
            },
            "back_camera": {
                "blueprint": SensorBlueprints.CAMERA_RGB,
                "location": (-5, 0, 1),
                "rotation": (0, 180, 0),
                "attributes": {},
            },
            "lidar": {
                "blueprint": SensorBlueprints.LIDAR_RANGE,
                "location": (0, 0, 2),
                "rotation": (0, 0, 0),
                "attributes": {},
            },
        },
    )

    context = AppContext(
        client=client,
        map=carla_map,
        sensor_map=sensor_map,
        sensor_data_queue=sensor_data_queue,
        actor_map={},
        ego_vehicle=ego_vehicle,
        folder_base_path=folder_base_path,
        images_base_path=images_base_path,
        controls_base_path=controls_base_path,
    )
    print("App env: ", context)
    # TODO: How to handle config variables from the command line like save folder?
    # Should it be part of the CarlaSimulationEnvironment object?
    game_loop_segment(context, [spectator_follow_ego_vehicle_task, save_image_task])


if __name__ == "__main__":
    main()
