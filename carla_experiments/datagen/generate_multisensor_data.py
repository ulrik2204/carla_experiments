from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import carla

from carla_experiments.carla_utils.setup import (
    CarlaSimulationEnvironment,
    game_loop_environment,
    initialize_carla_with_vehicle_and_sensors,
)
from carla_experiments.carla_utils.types_carla_utils import SensorBlueprintCollection


class AppActorsMap(TypedDict):
    ...


class AppSensorMap(TypedDict):
    front_camera: carla.Sensor
    back_camera: carla.Sensor
    lidar: carla.Sensor


class AppSensorsDataMap(TypedDict):
    front_camera: carla.Image
    back_camera: carla.Image
    lidar: carla.LidarMeasurement


@dataclass
class AppSimulationEnvironment(
    CarlaSimulationEnvironment[AppActorsMap, AppSensorMap, AppSensorsDataMap]
):
    folder_base_path: Path
    images_base_path: Path
    controls_base_path: Path


def save_image_task(environment: AppSimulationEnvironment) -> None:
    sensor_data = environment.get_sensor_data()
    front_image = sensor_data["front_camera"]
    back_image = sensor_data["back_camera"]
    lidar_point_cloud = sensor_data["lidar"]
    print("Front image, frame: ", front_image, front_image.frame)
    print("Back image, frame: ", back_image, back_image.frame)
    print("Lidar point cloud, frame: ", lidar_point_cloud, lidar_point_cloud.frame)
    front_image.save_to_disk(
        f"{environment.images_base_path.as_posix()}/font_{front_image.frame:06d}.jpg"
    )
    back_image.save_to_disk(
        f"{environment.images_base_path.as_posix()}/back_{back_image.frame:06d}.jpg"
    )
    lidar_point_cloud.save_to_disk(
        f"{environment.images_base_path.as_posix()}/lidar_{back_image.frame:06d}.ply"
    )


def spectator_follow_ego_vehicle_task(environment: AppSimulationEnvironment) -> None:
    ego_vehicle = environment.ego_vehicle
    vehicle_transform = ego_vehicle.get_transform()
    spectator = environment.world.get_spectator()
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

    environment = initialize_carla_with_vehicle_and_sensors(
        map="Town04",
        sensor_config={
            "front_camera": {
                "blueprint": SensorBlueprintCollection.CAMERA_RGB,
                "location": (2, 0, 1),
                "rotation": (0, 0, 0),
                "attributes": {},
            },
            "back_camera": {
                "blueprint": SensorBlueprintCollection.CAMERA_RGB,
                "location": (-5, 0, 1),
                "rotation": (0, 180, 0),
                "attributes": {},
            },
            "lidar": {
                "blueprint": SensorBlueprintCollection.LIDAR_RANGE,
                "location": (0, 0, 2),
                "rotation": (0, 0, 0),
                "attributes": {},
            },
        },
        ego_vehicle_spawn_point=lambda spawn_points: spawn_points[0],
    )
    # print("Environment: ", environment)
    app_env = AppSimulationEnvironment(
        **environment.__dict__,
        folder_base_path=folder_base_path,
        images_base_path=images_base_path,
        controls_base_path=controls_base_path,
    )
    print("App env: ", app_env)
    # TODO: How to handle config variables from the command line like save folder?
    # Should it be part of the CarlaSimulationEnvironment object?
    game_loop_environment(app_env, [spectator_follow_ego_vehicle_task, save_image_task])


if __name__ == "__main__":
    main()
