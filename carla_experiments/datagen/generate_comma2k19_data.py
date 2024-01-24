import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import List, NamedTuple, Optional, Tuple, TypedDict

import carla
import numpy as np

from carla_experiments.carla_utils.constants import SensorBlueprints
from carla_experiments.carla_utils.setup import (
    BatchContext,
    batch,
    create_dataset,
    segment,
    setup_carla_client,
    setup_sensors,
)
from carla_experiments.carla_utils.spawn import (
    spawn_ego_vehicle,
    spawn_vehicle_bots,
    spawn_walker_bots,
)
from carla_experiments.carla_utils.types_carla_utils import BatchResult, SegmentResult
from carla_experiments.datagen.utils import (
    carla_location_to_ecef,
    euler_to_quaternion,
    frames_to_video,
)


class AppActorMap(TypedDict):
    vehicles: List[carla.Vehicle]
    walkers: List[Tuple[carla.Walker, carla.WalkerAIController]]


class AppSensorMap(TypedDict):
    front_camera: carla.Sensor


class AppSensorDataMap(TypedDict):
    front_camera: carla.Image


@dataclass
class AppSettings:
    frame_rate: int
    folder_base_path: Path
    images_intermediary_folder: Path
    global_pose_path: Path
    locations_intermediary_folder: Path
    rotations_intermediary_folder: Path
    delete_intermediary_files: bool


@dataclass
class AppContext(BatchContext[AppSensorMap, AppActorMap], AppSettings):
    ...


def update_vehicle_lights_task(context: AppContext, _: AppSensorDataMap) -> None:
    traffic_manager = context.client.get_trafficmanager()
    vehicles = context.actor_map["vehicles"]
    for vehicle in vehicles:
        traffic_manager.update_vehicle_lights(vehicle, True)


def save_data_task(context: AppContext, sensor_data_map: AppSensorDataMap) -> None:
    front_image = sensor_data_map["front_camera"]
    # radar_data = parse_radar_data(sensor_data_map["radar"])
    # imu_data = parse_imu_data(sensor_data_map["imu"])
    # gnss_data = parse_gnss_data(sensor_data_map["gnss"])
    # speed = calculate_vehicle_speed(context.ego_vehicle)
    frame = front_image.frame
    # front_image.timestamp  # TODO: use this for frame times?
    ego_vehicle = context.ego_vehicle
    vehicle_transform = ego_vehicle.get_transform()
    location = vehicle_transform.location
    # TODO: Do I need to convert Location to ECEF coordinates?
    location_np = carla_location_to_ecef(context.map, location)
    roatation_np = euler_to_quaternion(vehicle_transform.rotation)
    # waypoint = (
    #     context.client.get_world().get_map().get_waypoint(ego_vehicle.get_location())
    # )
    # TODO: Append images to an array instead of saving them to disk
    front_image.save_to_disk(
        f"{context.images_intermediary_folder.as_posix()}/{frame:06d}.jpg"
    )
    # np.save(f"{context.radar_base_path.as_posix()}/{frame:06d}.npy", radar_data)
    np.save(
        f"{context.global_pose_path.as_posix()}/location/{frame:06d}.npy", location_np
    )
    np.save(
        f"{context.global_pose_path.as_posix()}/rotation/{frame:06d}.npy", roatation_np
    )


def spectator_follow_ego_vehicle_task(context: AppContext, _: AppSensorDataMap) -> None:
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


class Folders(NamedTuple):
    folder_base_path: Path
    images_intermediary_folder: Path
    global_pose_path: Path
    location_intermediary_folder: Path
    rotation_intermediary_folder: Path


def create_folders_if_not_exists(
    base_path: Optional[Path] = None,
) -> Folders:
    timestamp_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_base_path = base_path or Path(f"output/{timestamp_string}")
    folder_base_path.mkdir(parents=True, exist_ok=True)
    images_intermediary_folder = folder_base_path / "images"
    images_intermediary_folder.mkdir(parents=True, exist_ok=True)
    global_pose_path = folder_base_path / "global_pose"
    global_pose_path.mkdir(parents=True, exist_ok=True)
    location_intermediary_folder = global_pose_path / "location"
    location_intermediary_folder.mkdir(parents=True, exist_ok=True)
    rotation_intermediary_folder = global_pose_path / "rotation"
    rotation_intermediary_folder.mkdir(parents=True, exist_ok=True)
    return Folders(
        folder_base_path,
        images_intermediary_folder,
        global_pose_path,
        location_intermediary_folder,
        rotation_intermediary_folder,
    )


def concatenate_array_files(folder_path: Path, output_file_path: Path):
    """
    Concatenates numpy arrays from files in a given folder and saves the result.

    Args:
    folder_path (str): The path to the folder containing the files.
    output_file (str): The path of the output file to save the concatenated array.
    """
    # Get all file names in the folder
    files = [f for f in folder_path.iterdir() if f.is_file()]

    # Sort files based on frame number
    files.sort(key=lambda f: int(os.path.splitext(f.name)[0]))

    # Load and concatenate arrays
    arrays = []
    for file in files:
        arrays.append(np.load(file))

    concatenated_array = np.concatenate(arrays, axis=0)

    # Save the concatenated array
    print("Saving concatenated array to: ", output_file_path)
    with_npy = output_file_path.with_suffix(".npy")
    np.save(with_npy, concatenated_array)
    with_npy.rename(output_file_path)


def on_exit(context: AppContext) -> None:
    # This function will take all the images and create a hevc video
    print("Segment done, collecting files...")
    frames_to_video(
        context.images_intermediary_folder,
        context.folder_base_path / "video.hevc",
        context.frame_rate,
    )
    # This function will also take all the global poses and concatenate into a single file
    concatenate_array_files(
        context.locations_intermediary_folder,
        context.global_pose_path / "frame_locations",
    )
    concatenate_array_files(
        context.rotations_intermediary_folder,
        context.global_pose_path / "frame_orientations",
    )

    if context.delete_intermediary_files:
        for file in context.locations_intermediary_folder.iterdir():
            file.unlink()
        for file in context.rotations_intermediary_folder.iterdir():
            file.unlink()
        for file in context.images_intermediary_folder.iterdir():
            file.unlink()
        context.locations_intermediary_folder.rmdir()
        context.rotations_intermediary_folder.rmdir()
        context.images_intermediary_folder.rmdir()


@segment(frame_duration=20 * 30)  # 30 minutes (20 fps, 30s)
def stroll_segment(context: AppContext) -> SegmentResult:
    spawn_point = context.map.get_spawn_points()[1]
    context.ego_vehicle.set_transform(spawn_point)
    return {
        "tasks": [
            spectator_follow_ego_vehicle_task,
            save_data_task,
            update_vehicle_lights_task,
        ],
        "on_exit": on_exit,
    }


@batch
def first_batch(settings: AppSettings) -> BatchResult:
    # save_folder,

    client = setup_carla_client("Town04", frame_rate=settings.frame_rate)
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
        map=carla_map,
        sensor_map=sensor_map,
        sensor_data_queue=sensor_data_queue,
        actor_map={"vehicles": vehicle_bots, "walkers": walker_bots},
        ego_vehicle=ego_vehicle,
        frame_rate=settings.frame_rate,
        folder_base_path=settings.folder_base_path,
        images_intermediary_folder=settings.images_intermediary_folder,
        global_pose_path=settings.global_pose_path,
        locations_intermediary_folder=settings.locations_intermediary_folder,
        rotations_intermediary_folder=settings.rotations_intermediary_folder,
        delete_intermediary_files=True,
    )
    print("App env: ", context)
    # print("Starting game loop")
    return {
        "context": context,
        "segments": [stroll_segment],
        "on_exit": None,
    }


def main():
    batches = [first_batch]
    (
        folder_base_path,
        images_intermediary_folder,
        global_pose_path,
        location_intermediary_folder,
        rotation_intermediary_folder,
    ) = create_folders_if_not_exists()
    settings = AppSettings(
        frame_rate=20,
        folder_base_path=folder_base_path,
        images_intermediary_folder=images_intermediary_folder,
        global_pose_path=global_pose_path,
        locations_intermediary_folder=location_intermediary_folder,
        rotations_intermediary_folder=rotation_intermediary_folder,
        delete_intermediary_files=True,
    )
    create_dataset(batches, settings)


if __name__ == "__main__":
    main()
