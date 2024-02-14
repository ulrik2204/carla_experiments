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
        if os.path.exists(filepath):
            raise ValueError(f"File {filepath} already exists")
        # Progress denotes the index of the last completed task
        self.progress = -1
        self.tasklist = tasklist
        self.filepath = filepath
        if os.path.exists(filepath):
            print("File exists resuming from file.")
            with open(filepath, "r") as f:
                index = len(f.readlines())
                if tasklist[index] == f.readlines()[-1]:
                    self.progress = index
                else:
                    raise ValueError(
                        f"Tasks in file do not match tasklist. {tasklist[index]} != {f.readlines()[-1]}"
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
    print("Saving concatenated array to: ", output_file_path)
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


start_id = 0


def _generate_segment_id():
    global start_id
    start_id += 1
    return str(start_id)


def generate_stroll_segment(batch_segment_id: int):
    segment_id = _generate_segment_id()
    print("segment_id: ", segment_id)
    # 20 Hz for 60 seconds = 1200 frames
    frame_duration = 20 * 60

    def stroll_segment(context: AppContext) -> SegmentResult:
        print("Segment setting spawn point")
        len_points = len(context.map.get_spawn_points())
        spawn_point = context.map.get_spawn_points()[batch_segment_id % len_points]
        context.ego_vehicle.set_transform(spawn_point)
        print("Done setting spawn point")
        context.client.get_world().get_spectator().set_location(
            context.ego_vehicle.get_transform().location + carla.Location(z=2)
        )

        return {
            "tasks": [
                spectator_follow_ego_vehicle_task,
                save_data_task,
                update_vehicle_lights_task,
            ],
            "options": {
                "on_segment_end": on_segment_end,
            },
        }

    return segment(frame_duration=frame_duration, segment_base_folder=segment_id)(
        stroll_segment
    )


def configure_traffic_lights(world: carla.World):
    actors = world.get_actors().filter("traffic.traffic_light")
    for actor in actors:
        if isinstance(actor, carla.TrafficLight):
            actor.set_state(carla.TrafficLightState.Green)
            actor.set_green_time(5)
            actor.set_yellow_time(1)
            actor.set_red_time(5)


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
                    "attributes": {"image_size_x": "1164", "image_size_y": "874"},
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
        print("App env: ", context)
        # print("Starting game loop")
        generated_segments = [generate_stroll_segment(i) for i in range(200)]
        return {
            "context": context,
            "segments": generated_segments,
            "options": {},
        }

    return batch(batch_path)(first_batch)


def choose_chunks(
    all_chunks: Dict[str, List[Tuple[str, DecoratedBatch]]],
    tasklist: List[str],
    progress: int,
):
    if progress == len(tasklist) - 1:
        return {}
    if progress == -1:
        return all_chunks
    chunks: Dict[str, List[Tuple[str, DecoratedBatch]]] = {}
    rest_index = progress + 1
    rest_tasks = tasklist[rest_index:]
    for chunk_name, batches in all_chunks.items():
        for batch_name, batch_fn in batches:
            if batch_name in rest_tasks:
                if chunk_name not in chunks:
                    chunks[chunk_name] = []
                chunks[chunk_name].append((batch_name, batch_fn))
    return chunks


@click.command()
@click.option("--root-folder", type=str, default=None)
@click.option("--progress-file", type=str, default=None)
def main(root_folder: Optional[str], progress_file: Optional[str]):
    # Comma2k19 called this each batch by the date
    # TODO: Change back to Town01
    this_time = datetime.now().strftime("%m-%d_%H-%M")
    tasklist = [
        "batch1",
        "batch2",
        "batch3",
        "batch4",
        "batch5",
        "batch6",
        "batch7",
        "batch8",
        "batch9",
        "batch10",
    ]
    batch1 = create_batch("Town01", "batch1")
    batch2 = create_batch("Town02", "batch2")
    batch3 = create_batch("Town03", "batch3")
    batch4 = create_batch("Town04", "batch4")
    batch5 = create_batch("Town06", "batch5")
    batch6 = create_batch("Town07", "batch6")
    batch7 = create_batch("Town10HD", "batch7")
    batch8 = create_batch("Town04", "batch8")
    batch9 = create_batch("Town06", "batch9")
    batch10 = create_batch("Town10HD", "batch10")
    used_progress_file = (
        progress_file if progress_file is not None else f"progress-{this_time}.txt"
    )
    progress_handler = ProgressHandler(used_progress_file, tasklist)
    all_chunks = {
        "Chunk_1": [
            ("batch1", batch1),
            ("batch2", batch2),
            ("batch3", batch3),
            ("batch4", batch4),
            ("batch5", batch5),
        ],
        "Chunk_2": [
            ("batch6", batch6),
            ("batch7", batch7),
            ("batch8", batch8),
            ("batch9", batch9),
            ("batch10", batch10),
        ],
    }
    chunks = choose_chunks(all_chunks, tasklist, progress_handler.get_progress())
    # chunks = {"Chunk_1": [batch1]}
    if root_folder is None:
        base_path = Path("./output") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        base_path = Path(root_folder)

    settings = AppSettings(frame_rate=20)
    for chunk, batches in chunks.items():
        print("--- Creating", chunk, "---")
        used_batches = [batch_fn for _, batch_fn in batches]
        create_dataset(
            used_batches,
            base_path / chunk,
            settings,
            on_batch_end=progress_handler.update_progress,
        )


if __name__ == "__main__":
    main()
