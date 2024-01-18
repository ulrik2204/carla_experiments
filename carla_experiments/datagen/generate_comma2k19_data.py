import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import List, Tuple, TypedDict

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
from carla_experiments.carla_utils.spawn import (
    spawn_ego_vehicle,
    spawn_vehicle_bots,
    spawn_walker_bots,
)


class AppActorMap(TypedDict):
    # TODO: I need to explicitly allow this otherwise I cannot destroy them
    vehicles: List[carla.Vehicle]
    walkers: List[Tuple[carla.Walker, carla.WalkerAIController]]


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
class AppContext(CarlaContext[AppSensorMap, AppActorMap]):
    folder_base_path: Path
    images_base_path: Path
    radar_base_path: Path
    other_data_base_path: Path


def _save_dict_as_json(data: dict, path: Path):
    with path.open("w") as file:
        json.dump(data, file)


def update_vehicle_lights_task(context: AppContext, _: AppSensorDataMap) -> None:
    traffic_manager = context.client.get_trafficmanager()
    vehicles = context.actor_map["vehicles"]
    for vehicle in vehicles:
        traffic_manager.update_vehicle_lights(vehicle, True)


def save_data_task(context: AppContext, sensor_data_map: AppSensorDataMap) -> None:
    return
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
    return
    ego_vehicle = context.ego_vehicle
    vehicle_transform = ego_vehicle.get_transform()
    spectator = context.client.get_world().get_spectator()
    spectator_transform = spectator.get_transform()
    spectator_transform.location = vehicle_transform.location + carla.Location(z=2)
    spectator_transform.rotation.yaw = vehicle_transform.rotation.yaw
    spectator.set_transform(spectator_transform)


def configure_traffic_manager(
    traffic_manager: carla.TrafficManager,
    ego_vehicle: carla.Vehicle,
    vehicle_bots: List[carla.Vehicle],
) -> None:
    traffic_manager.set_random_device_seed(42)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_respawn_dormant_vehicles(True)
    # traffic_manager.set_hybrid_physics_mode(True)
    # traffic_manager.set_hybrid_physics_radius(70)

    for bot in vehicle_bots:
        traffic_manager.ignore_lights_percentage(bot, 5)
        traffic_manager.ignore_signs_percentage(bot, 5)
        traffic_manager.ignore_walkers_percentage(bot, 1)
        traffic_manager.vehicle_percentage_speed_difference(
            bot, random.randint(-30, 30)
        )
        traffic_manager.random_left_lanechange_percentage(bot, random.randint(0, 60))
        traffic_manager.random_right_lanechange_percentage(bot, random.randint(0, 60))


def create_folders_if_not_exists():
    timestamp_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_base_path = Path(f"output/{timestamp_string}")
    folder_base_path.mkdir(parents=True, exist_ok=True)
    images_base_path = folder_base_path / "images"
    images_base_path.mkdir(parents=True, exist_ok=True)
    radar_base_path = folder_base_path / "radar"
    radar_base_path.mkdir(parents=True, exist_ok=True)
    other_data_base_path = folder_base_path / "other"
    other_data_base_path.mkdir(parents=True, exist_ok=True)
    return folder_base_path, images_base_path, radar_base_path, other_data_base_path


def main():
    # save_folder,
    (
        folder_base_path,
        images_base_path,
        radar_base_path,
        other_data_base_path,
    ) = create_folders_if_not_exists()

    client = setup_carla_client("Town04")
    # client = setup_carla_client("Town10HD")
    world = client.get_world()
    carla_map = world.get_map()
    ego_vehicle = spawn_ego_vehicle(
        world, autopilot=True, spawn_point=carla_map.get_spawn_points()[0]
    )
    world.set_pedestrians_cross_factor(0.1)
    sensor_data_queue = Queue()
    # TODO: Check sensor positions
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
    print("spawning vehicles")
    vehicle_bots = spawn_vehicle_bots(world, 10)
    print("spawning bots")
    # TODO: Spawning walkers is not working, check generate_traffic example
    walker_bots = spawn_walker_bots(world, 15)
    print("configuring traffic manager")
    traffic_manager = client.get_trafficmanager()
    configure_traffic_manager(traffic_manager, ego_vehicle, vehicle_bots)

    # client.get_trafficmanager().set_global_distance_to_leading_vehicle()

    context = AppContext(
        client=client,
        sensor_map=sensor_map,
        sensor_data_queue=sensor_data_queue,
        actor_map={"vehicles": vehicle_bots, "walkers": walker_bots},
        ego_vehicle=ego_vehicle,
        folder_base_path=folder_base_path,
        images_base_path=images_base_path,
        radar_base_path=radar_base_path,
        other_data_base_path=other_data_base_path,
    )
    print("App env: ", context)
    print("Starting game loop")
    game_loop(
        context,
        [spectator_follow_ego_vehicle_task, save_data_task, update_vehicle_lights_task],
    )


if __name__ == "__main__":
    main()
