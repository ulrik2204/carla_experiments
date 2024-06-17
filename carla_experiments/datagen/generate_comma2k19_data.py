import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import List, Optional, Tuple, TypedDict

import carla
import click
import numpy as np
from PIL import Image

from carla_experiments.carla_utils.constants import SensorBlueprints
from carla_experiments.carla_utils.setup import (
    BatchContext,
    create_segment,
    generate_segment_dataset,
    setup_carla_client,
)
from carla_experiments.carla_utils.spawn import (
    spawn_ego_vehicle,
    spawn_sensors,
    spawn_vehicle_bots,
    spawn_walker_bots,
)
from carla_experiments.carla_utils.types_carla_utils import Segment, SegmentConfigResult
from carla_experiments.common.position_and_rotation import (
    carla_location_to_ecef,
    carla_rotation_to_ecef_frd_quaternion,
)
from carla_experiments.datagen.generate_comma3x_data import ProgressHandler
from carla_experiments.datagen.utils import mp4_to_hevc, pil_images_to_mp4


class AppActorMap(TypedDict):
    vehicles: List[carla.Vehicle]
    walkers: List[Tuple[carla.WalkerAIController, carla.Walker]]


class AppSensorMap(TypedDict):
    front_camera: carla.Sensor


class AppSensorDataMap(TypedDict):
    front_camera: carla.Image


@dataclass
class AppSettings:
    frame_rate: int
    client: carla.Client


class DataDict(TypedDict):
    location: List[np.ndarray]
    rotation: List[np.ndarray]
    images: List[Image.Image]
    carla_location: List[np.ndarray]
    carla_rotation: List[np.ndarray]


@dataclass
class AppContext(BatchContext[AppSensorMap, AppActorMap], AppSettings):
    data_dict: DataDict


def save_data_task(context: AppContext, sensor_data_map: AppSensorDataMap):
    front_image = sensor_data_map["front_camera"]
    ego_vehicle = context.ego_vehicle
    vehicle_transform = ego_vehicle.get_transform()
    location = vehicle_transform.location
    rotation = vehicle_transform.rotation
    location_np = carla_location_to_ecef(context.map, location)

    rotation_np = carla_rotation_to_ecef_frd_quaternion(context.map, rotation)
    ego_vehicle.get_angular_velocity()
    image = carla_image_to_pil_image(front_image)  # .transpose(Image.FLIP_LEFT_RIGHT)
    context.data_dict["images"].append(image)
    context.data_dict["location"].append(location_np)
    context.data_dict["rotation"].append(rotation_np)
    context.data_dict["carla_location"].append(
        np.array([location.x, location.y, location.z])
    )
    context.data_dict["carla_rotation"].append(
        np.array([rotation.pitch, rotation.yaw, rotation.roll])
    )


def spectator_follow_ego_vehicle_task(
    context: AppContext, sensor_data_map: AppSensorDataMap
) -> None:
    ego_vehicle = context.ego_vehicle
    vehicle_transform = ego_vehicle.get_transform()
    spectator = context.client.get_world().get_spectator()
    spectator_transform = spectator.get_transform()
    spectator_transform.location = vehicle_transform.location + carla.Location(z=2)
    spectator_transform.rotation = vehicle_transform.rotation
    spectator.set_transform(spectator_transform)


def configure_traffic_manager(
    traffic_manager: carla.TrafficManager,
    ego_vehicle: carla.Vehicle,
    vehicle_bots: List[carla.Vehicle],
) -> None:
    # traffic_manager.set_random_device_seed(42)
    # traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_respawn_dormant_vehicles(True)
    traffic_manager.set_boundaries_respawn_dormant_vehicles(25, 700)
    traffic_manager.set_route(ego_vehicle, ["Straight"] * 100000)

    # traffic_manager.set_desired_speed(ego_vehicle, 50 / 3.6)  # 50 km/h
    # traffic_manager.set_hybrid_physics_mode(True)
    # traffic_manager.set_hybrid_physics_radius(70)

    for bot in vehicle_bots:
        traffic_manager.ignore_lights_percentage(bot, 5)
        traffic_manager.ignore_signs_percentage(bot, 5)
        traffic_manager.ignore_walkers_percentage(bot, 1)
        traffic_manager.vehicle_percentage_speed_difference(
            bot, random.randint(-30, 30)
        )
        traffic_manager.random_left_lanechange_percentage(bot, random.randint(1, 60))
        traffic_manager.random_right_lanechange_percentage(bot, random.randint(1, 60))
        traffic_manager.update_vehicle_lights(bot, True)


def save_stacked_arrays(arrays: List[np.ndarray], output_file_path: Path):
    concatenated_array = np.vstack(arrays)
    with_npy = output_file_path.with_suffix(".npy")
    np.save(with_npy, concatenated_array)
    with_npy.rename(output_file_path)


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
    with_npy = output_file_path.with_suffix(".npy")
    np.save(with_npy, concatenated_array)
    with_npy.rename(output_file_path)


def carla_image_to_pil_image(image: carla.Image) -> Image.Image:
    array = np.frombuffer(np.copy(image.raw_data), dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))  # to BGRA image
    array = array[:, :, :3][:, :, ::-1]  # Convert to RGB
    return Image.fromarray(array)


def on_segment_end(context: AppContext, save_files_base_path: Path) -> None:
    # This function will take all the images and create a hevc video
    print("Saving video...")
    save_files_base_path.mkdir(parents=True, exist_ok=True)
    mp4_path = save_files_base_path / "video.mp4"

    pil_images_to_mp4(
        context.data_dict["images"],
        mp4_path.as_posix(),
        context.frame_rate,
    )
    mp4_to_hevc(mp4_path, save_files_base_path / "video.hevc")
    mp4_path.unlink()  # Remove the mp4 file

    # Saving positions and orientations

    print("Saving global poses...")
    # This function will also take all the global poses and concatenate into a single file
    global_pose_path = save_files_base_path / "global_pose"
    global_pose_path.mkdir(parents=True, exist_ok=True)
    save_stacked_arrays(
        context.data_dict["location"],
        global_pose_path / "frame_positions",
    )
    save_stacked_arrays(
        context.data_dict["rotation"],
        global_pose_path / "frame_orientations",
    )
    save_stacked_arrays(
        context.data_dict["carla_location"], global_pose_path / "carla_positions"
    )
    save_stacked_arrays(
        context.data_dict["carla_location"], global_pose_path / "carla_orientations"
    )


def configure_traffic_lights(world: carla.World):
    actors = world.get_actors().filter("traffic.traffic_light")
    for actor in actors:
        if isinstance(actor, carla.TrafficLight):
            actor.set_state(carla.TrafficLightState.Green)
            actor.set_green_time(5)
            actor.set_yellow_time(1)
            actor.set_red_time(5)


def generate_simple_segment(map: str, segment_path: Path, segment_nr_in_map: int):
    def simple_segment_config(settings: AppSettings) -> SegmentConfigResult:

        # client = setup_carla_client("Town10HD")
        client = settings.client
        if map not in client.get_world().get_map().name:
            print("Loading new map", map, "...")
            client.load_world(map, reset_settings=False)
            client.reload_world(reset_settings=False)
            time.sleep(5)
        else:
            print("Using previously loaded map", map)
        world = client.get_world()
        world.tick()
        carla_map = world.get_map()
        spawn_points = carla_map.get_spawn_points()
        len_spawn_points = len(spawn_points)
        spawn_point_index = segment_nr_in_map % len_spawn_points
        used_spawn_point = spawn_points[spawn_point_index]
        rotation = used_spawn_point.rotation
        location = used_spawn_point.location
        print("Generating in map", map)
        print(
            "Using spawn point",
            spawn_point_index,
            "of ",
            len_spawn_points,
            "with coords",
            f"({location.x:.2f}, {location.y:.2f}, {location.z:.2f})",
            "and rotation",
            f"({rotation.roll:.2f}, {rotation.pitch:.2f}, {rotation.yaw:.2f})",
        )
        spawn_point = spawn_points[spawn_point_index]
        ego_vehicle = spawn_ego_vehicle(world, autopilot=True, spawn_point=spawn_point)
        world.set_pedestrians_cross_factor(0.1)
        sensor_data_queue = Queue()
        # TODO: Check sensor positions
        sensor_map = spawn_sensors(
            world,
            ego_vehicle,
            sensor_data_queue=sensor_data_queue,
            return_sensor_map_type=AppSensorMap,
            sensor_config={
                "front_camera": {
                    "blueprint": SensorBlueprints.CAMERA_RGB,
                    "location": (2, 0, 1),
                    "rotation": (0, 0, 0),
                    "attributes": {"image_size_x": "1164", "image_size_y": "874"},
                },
            },
        )
        vehicle_bot_spawn_points = spawn_points.copy()
        vehicle_bot_spawn_points.pop(spawn_point_index)
        vehicle_bots = spawn_vehicle_bots(
            world, 10, accessible_spawn_points=vehicle_bot_spawn_points
        )
        walker_bots = spawn_walker_bots(world, 15)
        traffic_manager = client.get_trafficmanager()
        configure_traffic_manager(traffic_manager, ego_vehicle, vehicle_bots)
        configure_traffic_lights(world)

        # client.get_trafficmanager().set_global_distance_to_leading_vehicle()
        data_dict: DataDict = {
            "location": [],
            "rotation": [],
            "images": [],
            "carla_location": [],
            "carla_rotation": [],
        }
        actors = world.get_actors()
        print("There should be 15 walkers, 10 vehicles and 1 ego vehicle.")
        number_of_actors = len(actors)
        number_of_vehicles = len(world.get_actors().filter("vehicle.*"))
        number_of_walkers = len(world.get_actors().filter("walker.*"))
        print(f"{number_of_actors=}, {number_of_vehicles=}, {number_of_walkers=}")

        context = AppContext(
            client=client,
            map=carla_map,
            sensor_map=sensor_map,
            sensor_data_queue=sensor_data_queue,
            actor_map={"vehicles": vehicle_bots, "walkers": walker_bots},
            ego_vehicle=ego_vehicle,
            frame_rate=settings.frame_rate,
            data_dict=data_dict,
        )
        return {
            "context": context,
            "tasks": [
                spectator_follow_ego_vehicle_task,
                save_data_task,
            ],
            "options": {
                "on_segment_end": on_segment_end,
                "cleanup_actors": True,
            },
        }

    frame_duration = 20 * 60  # 20 Hz for 60 seconds = 1200 frames
    return create_segment(frame_duration, segment_path, simple_segment_config)


def get_some(i, j, k):
    return f"Chunk_{i}/Batch_{j}/Segment_{k}"


def choose_segments(all_segments: List[Segment], progress: int) -> List[Segment]:
    print("progress", progress)
    print("all_segments", len(all_segments))
    if progress == len(all_segments):
        return []
    if progress == -1:
        return all_segments
    rest_start_index = progress + 1
    return all_segments[rest_start_index:]


def create_all_segments(
    base_path: Path,
    segments_relative_paths: List[str],
    towns: List[str],
    segments_per_town=200,
):
    allsegments = []
    segments_relative_paths = [
        (base_path / item).as_posix() for item in segments_relative_paths
    ]
    for i, town in enumerate(towns):
        subtasks = segments_relative_paths[
            i * segments_per_town : (i + 1) * segments_per_town
        ]
        subsegments = [
            generate_simple_segment(town, Path(item), i)
            for i, item in enumerate(subtasks, 1)
        ]
        allsegments.extend(subsegments)
    return allsegments


def after_segment_end(progress_handler: ProgressHandler):
    progress_handler.update_progress()
    if not progress_handler.is_finished():
        # Restart the script after each segment
        sys.exit(1)


@click.command()
@click.option("--root-folder", type=str, default=None)
@click.option("--progress-file", type=str, default=None)
def main(root_folder: Optional[str], progress_file: Optional[str]):
    # Comma2k19 called this each batch by the date
    # TODO: Change back to Town01
    this_time = datetime.now().strftime("%m-%d_%H-%M")
    used_progress_file = (
        progress_file if progress_file is not None else f"progress-{this_time}.txt"
    )
    if root_folder is None:
        base_path = Path("./output") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        base_path = Path(root_folder)
    number_of_segments = 2019
    segments_relative_paths = [
        f"Chunk_{i//100 + 1}/Batch_{i//10 + 1}/Segment_{i}"
        for i in range(0, number_of_segments)
    ]
    print(
        "segment_save_path_list",
        segments_relative_paths[-1],
        "len",
        len(segments_relative_paths),
    )
    progress_handler = ProgressHandler(used_progress_file, segments_relative_paths)
    # Original townlist
    # townlist = [
    #     "Town01",
    #     "Town02",
    #     "Town03",
    #     "Town04",
    #     "Town06",
    #     "Town07",
    #     "Town10HD",
    #     "Town04",
    #     "Town06",
    #     "Town07",
    # ]
    townlist = ["Town04", "Town06"]
    segments_per_town = 1010  # Will be 1010 of Town04 and 1009 of Town06
    all_segments = create_all_segments(
        base_path, segments_relative_paths, townlist, segments_per_town
    )
    chosen_segments = choose_segments(all_segments, progress_handler.get_progress())
    print(
        "Starting from segment",
        segments_relative_paths[progress_handler.get_progress() + 1],
    )
    print("Chosen segments", len(chosen_segments))
    if len(chosen_segments) == 0:
        print("No segments left to run.")
        return
    # chunks = {"Chunk_1": [batch1]}
    carla_client = setup_carla_client()
    settings = AppSettings(frame_rate=20, client=carla_client)
    generate_segment_dataset(
        chosen_segments,
        settings,
        after_segment_end=lambda _: after_segment_end(progress_handler),
    )


if __name__ == "__main__":
    main()
