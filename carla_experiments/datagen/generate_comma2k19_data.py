import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import carla
import click
import numpy as np
from PIL import Image

from carla_experiments.carla_utils.constants import SensorBlueprints
from carla_experiments.carla_utils.setup import (
    BatchContext,
    batch,
    create_dataset,
    create_segment,
    generate_segment_dataset,
    segment,
    setup_carla_client,
    setup_sensors,
)
from carla_experiments.carla_utils.spawn import (
    spawn_ego_vehicle,
    spawn_vehicle_bots,
    spawn_walker_bots,
)
from carla_experiments.carla_utils.types_carla_utils import (
    BatchResult,
    DecoratedBatch,
    FullSegment,
    FullSegmentConfigResult,
    SegmentResult,
)
from carla_experiments.datagen.utils import (
    carla_location_to_ecef,
    euler_to_quaternion2,
    mp4_to_hevc,
    pil_images_to_mp4,
)


class ProgressHandler:

    def __init__(self, filepath: str, tasklist: List[str]) -> None:
        # Progress denotes the index of the last completed task
        self.progress = -1
        self.tasklist = tasklist
        self.filepath = filepath
        if os.path.exists(filepath):
            print("File exists resuming from file.")
            print("filepath", filepath)
            with open(filepath, "r") as f:
                els = f.readlines()
                index = len(els) - 1
                task = tasklist[index].strip()
                el = els[index].strip()
                if task == el:
                    self.progress = index
                else:
                    raise ValueError(
                        f"Tasks in file do not match tasklist. {task} != {el}"
                    )

    def get_progress(self) -> int:
        return self.progress

    def update_progress(self) -> None:
        self.progress += 1
        with open(self.filepath, "a") as f:
            f.write(f"{self.tasklist[self.progress]}\n")


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
    client: carla.Client


class DataDict(TypedDict):
    location: List[np.ndarray]
    rotation: List[np.ndarray]
    images: List[Image.Image]


@dataclass
class AppContext(BatchContext[AppSensorMap, AppActorMap], AppSettings):
    data_dict: DataDict


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
    # front_image.timestamp  # TODO: use this for frame times?
    ego_vehicle = context.ego_vehicle
    vehicle_transform = ego_vehicle.get_transform()
    location = vehicle_transform.location
    rotation = vehicle_transform.rotation
    location_np = carla_location_to_ecef(context.map, location)
    # location_np = np.array([location.x, location.y, location.z])
    rotation_np = euler_to_quaternion2(context.map, rotation)
    # rotation_np = np.array([rotation.pitch, rotation.yaw, rotation.roll])
    # frame = _generate_frame_id()
    # print(f"before add [frame {frame}]", len(context.data_dict["images"]))
    image = carla_image_to_pil_image(front_image)  # .transpose(Image.FLIP_LEFT_RIGHT)
    context.data_dict["images"].append(image)
    context.data_dict["location"].append(location_np)
    context.data_dict["rotation"].append(rotation_np)
    # print(f"after add [frame {frame}]", len(context.data_dict["images"]))
    # return {
    #     "location": {
    #         str(frame): location_np,
    #     },
    #     "rotation": {
    #         str(frame): rotation_np,
    #     },
    #     "images": {
    #         str(frame): front_image,
    #     },
    # }


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
    traffic_manager.set_random_device_seed(42)
    # traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_respawn_dormant_vehicles(True)
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
    # Resetting the data_dict after each segment
    context.data_dict = {"location": [], "rotation": [], "images": []}


def configure_traffic_lights(world: carla.World):
    actors = world.get_actors().filter("traffic.traffic_light")
    for actor in actors:
        if isinstance(actor, carla.TrafficLight):
            actor.set_state(carla.TrafficLightState.Green)
            actor.set_green_time(5)
            actor.set_yellow_time(1)
            actor.set_red_time(5)


def generate_simple_segment(map: str, segment_path: Path, spawn_point_idx: int):
    def simple_segment_config(settings: AppSettings) -> FullSegmentConfigResult:

        # client = setup_carla_client("Town10HD")
        same_map = True
        client = settings.client
        if map not in client.get_world().get_map().name:
            client.load_world(map, reset_settings=False)
            same_map = False
        world = client.get_world()
        world.tick()
        carla_map = world.get_map()
        spawn_points = carla_map.get_spawn_points()
        len_spawn_points = len(spawn_points)
        spawn_point_index = spawn_point_idx % len_spawn_points
        used_spawn_point = spawn_points[spawn_point_index]
        print("Generating in map", map)
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
        spawn_point = spawn_points[spawn_point_index]
        ego_vehicle = spawn_ego_vehicle(world, autopilot=True, spawn_point=spawn_point)
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
                    "attributes": {"image_size_x": "1164", "image_size_y": "874"},
                },
            },
        )

        vehicle_bots = spawn_vehicle_bots(world, 10)
        walker_bots = spawn_walker_bots(world, 15)
        traffic_manager = client.get_trafficmanager()
        configure_traffic_manager(traffic_manager, ego_vehicle, vehicle_bots)
        configure_traffic_lights(world)

        # client.get_trafficmanager().set_global_distance_to_leading_vehicle()
        data_dict: DataDict = {"location": [], "rotation": [], "images": []}

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
                update_vehicle_lights_task,
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


def choose_segments(
    all_segments: List[FullSegment], progress: int
) -> List[FullSegment]:
    if progress == len(all_segments) - 1:
        return []
    if progress == -1:
        return all_segments
    rest_start_index = progress + 1
    return all_segments[rest_start_index:]


def create_all_segments(tasklist: List[str]):
    town01_segments = [
        generate_simple_segment("Town01", Path(item), i)
        for i, item in enumerate(tasklist[:200], 1)
    ]
    town02_segments = [
        generate_simple_segment("Town02", Path(item), i)
        for i, item in enumerate(tasklist[201:400], 1)
    ]
    town03_segments = [
        generate_simple_segment("Town03", Path(item), i)
        for i, item in enumerate(tasklist[401:600], 1)
    ]
    town04_segments = [
        generate_simple_segment("Town04", Path(item), i)
        for i, item in enumerate(tasklist[601:800], 1)
    ]
    town06_segments = [
        generate_simple_segment("Town06", Path(item), i)
        for i, item in enumerate(tasklist[801:1000], 1)
    ]
    town07_segments = [
        generate_simple_segment("Town07", Path(item), i)
        for i, item in enumerate(tasklist[1000:1201], 1)
    ]
    town10_segments = [
        generate_simple_segment("Town10HD", Path(item), i)
        for i, item in enumerate(tasklist[1201:1400], 1)
    ]
    town042_segments = [
        generate_simple_segment("Town04", Path(item), i)
        for i, item in enumerate(tasklist[1400:1601], 1)
    ]
    town062_segments = [
        generate_simple_segment("Town06", Path(item), i)
        for i, item in enumerate(tasklist[1600:1801], 1)
    ]
    town072_segments = [
        generate_simple_segment("Town07", Path(item), i)
        for i, item in enumerate(tasklist[1801:], 1)
    ]
    return (
        town01_segments
        + town02_segments
        + town03_segments
        + town04_segments
        + town06_segments
        + town07_segments
        + town10_segments
        + town042_segments
        + town062_segments
        + town072_segments
    )


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
    tasklist = [
        (base_path / f"Chunk_{i//100 + 1}/Batch_{i//10 + 1}/Segment_{i}").as_posix()
        for i in range(1, 2001)
    ]
    progress_handler = ProgressHandler(used_progress_file, tasklist)
    all_segments = create_all_segments(tasklist)
    chosen_segments = choose_segments(all_segments, progress_handler.get_progress())
    print("Starting from segment", tasklist[progress_handler.get_progress() + 1])
    # chunks = {"Chunk_1": [batch1]}
    carla_client = setup_carla_client()
    settings = AppSettings(frame_rate=20, client=carla_client)
    generate_segment_dataset(
        chosen_segments,
        settings,
        after_segment_end=lambda _: progress_handler.update_progress(),
    )


if __name__ == "__main__":
    main()
