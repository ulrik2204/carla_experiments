import sys
from pathlib import Path
from queue import Empty
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, TypeVar, Union

import carla
import numpy as np
from PIL import Image

from carla_experiments.carla_utils.types_carla_utils import (
    BatchContext,
    CarlaTask,
    FlexiblePath,
    SaveItems,
    Segment,
    SegmentConfig,
)

TSensorMap = TypeVar("TSensorMap", bound=Mapping[str, Any])
TSensorDataMap = TypeVar("TSensorDataMap", bound=Mapping[str, Any])
TActorMap = TypeVar("TActorMap", bound=Mapping[str, Any])


def _flexible_path_to_path(flexible_path: Optional[FlexiblePath]) -> Path:
    if flexible_path is None:
        return Path("")
    elif isinstance(flexible_path, str):
        return Path(flexible_path)
    elif isinstance(flexible_path, Path):
        return flexible_path
    else:
        raise ValueError(f"Unknown path type {type(flexible_path)}")


def setup_carla_client(map: Optional[str] = None, frame_rate: int = 20):
    """Creates the CARLA client at given port and
    sets the world to synchronous mode with given frame rate.
    Also sets the timeout to 30 seconds.

    Args:
        map (str): The map to load. E.g. "Town01"
        frame_rate (int, optional): Frame rate to run the simulation in fps. Defaults to 20.

    Returns:
        Client: The CARLA client object
    """
    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)
    if map is not None:
        client.load_world(map)
    world = client.get_world()
    client.get_trafficmanager().set_synchronous_mode(True)

    settings = world.get_settings()
    settings.synchronous_mode = True  # Enables synchronous mode
    settings.fixed_delta_seconds = 1.0 / frame_rate
    settings.actor_active_distance = 2000
    world.apply_settings(settings)
    return client


TContext = TypeVar("TContext", bound=BatchContext)


def _get_sensor_data_map(env: BatchContext, queue_timeout: float = 10):
    try:
        data_dict = {}
        while len(data_dict.keys()) < len(env.sensor_map.keys()):
            # # Don't wait for the opendrive sensor
            # if self._opendrive_tag and self._opendrive_tag not in data_dict.keys() \
            #         and len(self._sensors_objects.keys()) == len(data_dict.keys()) + 1:
            #     break

            sensor_id, data = env.sensor_data_queue.get(True, queue_timeout)
            data_dict[sensor_id] = data

    except Empty:
        # TODO: Maybe not throw exception?
        raise Exception("A sensor took too long to send their data")

    return data_dict


class StopSegment(Exception):
    pass


TSaveFileBasePath = TypeVar("TSaveFileBasePath", bound=Optional[Path])


def _stop_loop(
    context: TContext,
    save_files_base_path: TSaveFileBasePath,
    on_finished: Optional[Callable[[TContext, TSaveFileBasePath], None]] = None,
    cleanup_actors: bool = False,
):
    if on_finished is not None:
        on_finished(context, save_files_base_path)
    if cleanup_actors:
        print("Cleaning up actors...")
        stop_actors(context.actor_map)
        stop_actors(context.sensor_map)
        stop_actors(context.ego_vehicle)


def game_loop_segment(
    context: TContext,
    tasks: List[CarlaTask[TContext, TSensorDataMap]],
    on_finished: Optional[Callable[[TContext, TSaveFileBasePath], None]] = None,
    max_frames: Optional[int] = None,
    save_files_base_path: TSaveFileBasePath = None,
    cleanup_actors: bool = False,
):
    # print("Preparing segment in loop")
    # for _ in range(50):
    #     _get_sensor_data_map(context)
    #     context.client.get_world().tick()
    #     time.sleep(0.01)
    print("Starting segment in loop")
    frames = 0
    while True:
        try:  # in case of a crash, try to recover and continue
            if max_frames is not None and frames >= max_frames:
                print("Cleaning up actors...")
                _stop_loop(context, save_files_base_path, on_finished, cleanup_actors)
                break
            sensor_data_map = _get_sensor_data_map(context)
            for task in tasks:
                save_items = task(context, sensor_data_map)  # type: ignore
                if save_items is not None and save_files_base_path is not None:
                    save_items_to_file(save_files_base_path, save_items)
            # time.sleep(0.01)
            context.client.get_world().tick()
            frames += 1
        except KeyboardInterrupt:
            print("Cleaning up actors...")
            _stop_loop(context, save_files_base_path, None, cleanup_actors=True)
            sys.exit()


def stop_actors(
    actor: Union[
        carla.Actor, List[carla.Actor], Dict[str, carla.Actor], Tuple[carla.Actor, ...]
    ]
):
    try:

        if isinstance(actor, carla.Sensor):
            actor.stop()
            actor.destroy()
        elif isinstance(actor, carla.Actor):
            actor.destroy()
        elif isinstance(actor, dict):
            for a in actor.values():
                stop_actors(a)
            actor.clear()
        elif isinstance(actor, list) or isinstance(actor, tuple):
            for a in actor:
                stop_actors(a)
            if isinstance(actor, list):
                actor.clear()
        else:
            print("Unknown actor type", type(actor))
    except Exception as e:
        print("Error stopping actors", e)


TSettings = TypeVar("TSettings")


def save_items_to_file(base_path: Path, items: SaveItems):
    base_path.mkdir(parents=True, exist_ok=True)
    item_dicts = [(base_path, items)]
    files: List[Path] = []
    while True:
        used_base_path, used_items = item_dicts.pop(0)
        for path, value in used_items.items():
            path_to_save = used_base_path / path
            if isinstance(value, dict):
                path_to_save.mkdir(parents=True, exist_ok=True)
                item_dicts.append((path_to_save, value))
            elif isinstance(value, Image.Image):
                value.save(path_to_save.as_posix())
            elif isinstance(value, np.ndarray):
                np.save(path_to_save, value)
                files.append(path_to_save)
            elif isinstance(value, carla.Image):
                value.save_to_disk(path_to_save.as_posix())
                files.append(path_to_save)
            else:
                raise ValueError(f"Unknown save item type {type(value)}")
        if len(item_dicts) == 0:
            break


def create_segment(
    frame_duration: Optional[int],
    segment_base_folder: Optional[FlexiblePath],
    segment_config: SegmentConfig[TSettings, TContext, TSensorDataMap],
) -> Segment:
    """Creates a segment function that can be run in a loop.
    The segment is configued using the segment_config function.
    The segment_config function is passed the settings, context and sensor data map
    from the game loop.

    Args:
        frame_duration (Optional[int]): The durection of the segment in seconds.
        segment_base_folder (Optional[FlexiblePath]): The base folder to save the segment files in.
        segment_config (FullSegmentConfig[TSettings, TContext, TSensorDataMap]): The segment config function.

    Returns:
        Segment: The segment function.
    """

    def inner(settings: TSettings) -> None:
        segment_result = segment_config(settings)
        print("Starting segment")
        segment_path = _flexible_path_to_path(segment_base_folder)
        context = segment_result["context"]
        tasks = segment_result["tasks"]
        optionals = segment_result["options"]
        cleanup_actors = (
            optionals["cleanup_actors"] if "cleanup_actors" in optionals else False
        )

        on_segment_end = (
            optionals["on_segment_end"] if "on_segment_end" in optionals else None
        )

        game_loop_segment(
            context=context,
            tasks=tasks,
            on_finished=on_segment_end,
            max_frames=frame_duration,
            save_files_base_path=segment_path,
            cleanup_actors=cleanup_actors,
        )

    return inner


def generate_segment_dataset(
    segments: List[Segment],
    settings: TSettings,
    after_segment_end: Optional[Callable[[TSettings], None]] = None,
):
    """A function to run a list of segments.
    The function just loops through the segments and calls them with
    the settings.

    Args:
        segments (List[Segment]): The list of segments to be run.
        settings (TSettings): The application settings.
        after_segment_end (Optional[Callable[[TSettings], None]], optional):
            Function to be called after a segment is done. Defaults to None.
    """
    for segment in segments:
        segment(settings)
        if after_segment_end is not None:
            after_segment_end(settings)
