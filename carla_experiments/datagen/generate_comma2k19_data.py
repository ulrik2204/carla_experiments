import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import List, Optional, Tuple, TypedDict

import carla
import click
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
    # folder_base_path: Path
    # images_intermediary_folder: Path
    # global_pose_path: Path
    # locations_intermediary_folder: Path
    # rotations_intermediary_folder: Path
    delete_intermediary_files: bool


@dataclass
class AppContext(BatchContext[AppSensorMap, AppActorMap], AppSettings):
    ...


def update_vehicle_lights_task(
    context: AppContext, sensor_data_map: AppSensorDataMap
) -> None:
    traffic_manager = context.client.get_trafficmanager()
    vehicles = context.actor_map["vehicles"]
    for vehicle in vehicles:
        traffic_manager.update_vehicle_lights(vehicle, True)


def save_data_task(context: AppContext, sensor_data_map: AppSensorDataMap):
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
    frame = f"{front_image.frame:06d}"
    return {
        "location": {
            frame: location_np,
        },
        "rotation": {
            frame: roatation_np,
        },
        "images": {
            frame: front_image,
        },
    }


def spectator_follow_ego_vehicle_task(
    context: AppContext, sensor_data_map: AppSensorDataMap
) -> None:
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


def combine_array_files(folder_path: Path, output_file_path: Path):
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

    concatenated_array = np.vstack(arrays)

    # Save the concatenated array
    print("Saving concatenated array to: ", output_file_path)
    with_npy = output_file_path.with_suffix(".npy")
    np.save(with_npy, concatenated_array)
    with_npy.rename(output_file_path)


def on_segment_end(context: AppContext, save_files_base_path: Path) -> None:
    # This function will take all the images and create a hevc video
    print("Segment done, collecting files...")
    images_path = save_files_base_path / "images"
    locations_path = save_files_base_path / "location"
    rotations_path = save_files_base_path / "rotation"
    frames_to_video(
        images_path,
        save_files_base_path / "video.hevc",
        context.frame_rate,
    )
    # This function will also take all the global poses and concatenate into a single file
    global_pose_path = save_files_base_path / "global_pose"
    global_pose_path.mkdir(parents=True, exist_ok=True)
    combine_array_files(
        locations_path,
        global_pose_path / "frame_positions",
    )
    combine_array_files(
        rotations_path,
        global_pose_path / "frame_orientations",
    )

    if context.delete_intermediary_files:
        for file in locations_path.iterdir():
            file.unlink()
        for file in rotations_path.iterdir():
            file.unlink()
        for file in images_path.iterdir():
            file.unlink()
        locations_path.rmdir()
        rotations_path.rmdir()
        images_path.rmdir()


start_id = 0


def _generate_segment_id():
    global start_id
    start_id += 1
    return str(start_id)


def generate_stroll_segment():
    segment_id = _generate_segment_id()
    print("segment_id: ", segment_id)

    def stroll_segment(context: AppContext) -> SegmentResult:
        spawn_point = random.choice(context.map.get_spawn_points())
        context.ego_vehicle.set_transform(spawn_point)
        return {
            "tasks": [
                spectator_follow_ego_vehicle_task,
                save_data_task,
                update_vehicle_lights_task,
            ],
            "options": {"on_segment_end": on_segment_end},
        }

    return segment(frame_duration=20 * 30, segment_base_folder=segment_id)(
        stroll_segment
    )


def create_batch(map: str, batch_path: str):
    def first_batch(settings: AppSettings) -> BatchResult:
        # save_folder,

        client = setup_carla_client(map, frame_rate=settings.frame_rate)
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
            # folder_base_path=settings.folder_base_path,
            # images_intermediary_folder=settings.images_intermediary_folder,
            # global_pose_path=settings.global_pose_path,
            # locations_intermediary_folder=settings.locations_intermediary_folder,
            # rotations_intermediary_folder=settings.rotations_intermediary_folder,
            delete_intermediary_files=True,
        )
        print("App env: ", context)
        # print("Starting game loop")
        # TODO: Handle to not stop actors if it is a segment, but only at the end of a batch
        generated_segments = [generate_stroll_segment() for _ in range(3)]
        return {
            "context": context,
            "segments": generated_segments,
            "options": {},
        }

    return batch(batch_path)(first_batch)


@click.command()
@click.option("--root-folder", type=str, default=None)
def main(root_folder: Optional[str]):
    batch1 = create_batch(
        "Town01", "batch1"
    )  # Comma2k19 called this each batch by the date
    batch2 = create_batch("Town02", "batch2")
    batch3 = create_batch("Town03", "batch3")
    batch4 = create_batch("Town04", "batch4")
    chunks = {"Chunk_1": [batch1, batch2], "Chunk_2": [batch3, batch4]}
    if root_folder is None:
        base_path = Path("./output") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        base_path = Path(root_folder)

    settings = AppSettings(
        frame_rate=20,
        delete_intermediary_files=True,
    )
    for chunk, batches in chunks.items():
        print("--- Creating", chunk, "---")
        create_dataset(batches, base_path / chunk, settings)


if __name__ == "__main__":
    main()
