import sys
import time
from functools import wraps
from pathlib import Path
from queue import Empty, Queue
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import carla
import numpy as np
from PIL import Image

from carla_experiments.carla_utils.constants import AttributeDefaults, SensorBlueprints
from carla_experiments.carla_utils.spawn import spawn_sensor
from carla_experiments.carla_utils.types_carla_utils import (
    Batch,
    BatchContext,
    CarlaTask,
    DecoratedBatch,
    DecoratedSegment,
    FlexiblePath,
    FullSegment,
    FullSegmentConfig,
    SaveItems,
    Segment,
    SensorBlueprint,
    SensorConfig,
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


def _create_blueprint_changer(
    attributes: Mapping[str, str],
    defaults: Mapping[str, str],
):
    def inner(blueprint: carla.ActorBlueprint):
        for key, value in defaults.items():
            blueprint.set_attribute(key, value)
        for key, value in attributes.items():
            blueprint.set_attribute(key, value)
        return blueprint

    return inner


def _handle_sensor_setup(
    world: carla.World,
    ego_vehicle: carla.Vehicle,
    sensor_id: str,
    sensor_config: SensorConfig,
    sensor_data_queue: Queue,
) -> carla.Sensor:
    sensor_blueprint = sensor_config["blueprint"]
    attributes = sensor_config["attributes"]

    modify_camera_bp = _create_blueprint_changer(attributes, AttributeDefaults.CAMERA)
    modify_lidar_bp = _create_blueprint_changer(attributes, AttributeDefaults.LIDAR)
    modify_radar_bp = _create_blueprint_changer(attributes, AttributeDefaults.RADAR)
    modify_imu_bp = _create_blueprint_changer(attributes, AttributeDefaults.IMU)
    modify_gnss_bp = _create_blueprint_changer(attributes, AttributeDefaults.GNSS)

    modify_blueprint_fn_map: Dict[
        SensorBlueprint[Any],
        Optional[Callable[[carla.ActorBlueprint], carla.ActorBlueprint]],
    ] = {
        SensorBlueprints.CAMERA_RGB: modify_camera_bp,
        SensorBlueprints.CAMERA_DEPTH: modify_camera_bp,
        SensorBlueprints.CAMERA_DVS: modify_camera_bp,
        SensorBlueprints.CAMERA_SEMANTIC_SEGMENTATION: modify_camera_bp,
        SensorBlueprints.COLLISION: None,
        SensorBlueprints.GNSS: modify_gnss_bp,
        SensorBlueprints.IMU: modify_imu_bp,
        SensorBlueprints.LANE_INVASION: None,
        SensorBlueprints.OBSTACLE: None,
        SensorBlueprints.LIDAR_RANGE: modify_lidar_bp,
        SensorBlueprints.LIDAR_SEMANTIC_SEGMENTATION: modify_lidar_bp,
        SensorBlueprints.RADAR_RANGE: modify_radar_bp,
    }
    if sensor_blueprint in modify_blueprint_fn_map:
        modify_blueprint_fn = modify_blueprint_fn_map[sensor_blueprint]
    else:
        raise ValueError(f"Unknown sensor blueprint {sensor_blueprint}")

    sensor = spawn_sensor(
        world,
        sensor_blueprint,
        sensor_config["location"],
        sensor_config["rotation"],
        ego_vehicle,
        modify_blueprint_fn=modify_blueprint_fn,
        on_measurement_received=lambda data: sensor_data_queue.put((sensor_id, data)),
    )
    return sensor


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


def setup_sensors(
    world: carla.World,
    ego_vehicle: carla.Vehicle,
    sensor_data_queue: Queue,
    return_sensor_map_type: Type[TSensorMap],
    sensor_config: Mapping[str, SensorConfig],
) -> TSensorMap:
    sensor_map: Mapping[str, carla.Sensor] = {}
    for sensor_id, config in sensor_config.items():
        sensor = _handle_sensor_setup(
            world, ego_vehicle, sensor_id, config, sensor_data_queue
        )
        sensor_map[sensor_id] = sensor
    time.sleep(0.1)
    world.tick()
    return cast(return_sensor_map_type, sensor_map)


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
        if isinstance(actor, carla.Actor):
            carla.command.DestroyActor(actor)  # type: ignore
        elif isinstance(actor, carla.Sensor):
            actor.stop()
            carla.command.DestroyActor(actor)  # type: ignore
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


def create_batch_dataset(
    batches: List[DecoratedBatch[TSettings]],
    base_folder: FlexiblePath,
    settings: TSettings,
    on_batch_end: Optional[Callable[[], None]] = None,
    on_segment_end: Optional[Callable[[], None]] = None,
):
    # This function is used to create a dataset from a list of batches
    base_path = _flexible_path_to_path(base_folder)
    for batch in batches:
        batch(base_path, settings, on_segment_end=on_segment_end)
        if on_batch_end is not None:
            on_batch_end()


def batch(batch_folder: FlexiblePath):
    """Decorator for a batch function. A batch function is a function that
        sets up a CARLA client and world, spawns an ego vehicle, and sets up sensors.
        All segments in a batch share the same context created by the batch.



    Args:
        batch_folder (FlexiblePath): The folder to save the batch data in.

    """

    def decorator(func: Batch[TSettings]) -> DecoratedBatch[TSettings]:
        # This is a function to also be able to group the data generation into batches
        # Each batch is a collection of segments with certain world settings.
        @wraps(func)
        def inner(
            base_path: Path,
            settings: TSettings,
            on_segment_end: Optional[Callable[[], None]] = None,
        ) -> None:
            batch_result = func(settings)
            batch_path = _flexible_path_to_path(batch_folder)
            full_base_path = base_path / batch_path
            for segment in batch_result["segments"]:
                segment(batch_result["context"], full_base_path)
                if on_segment_end is not None:
                    on_segment_end()
            optionals = batch_result["options"]
            on_exit = optionals["on_batch_end"] if "on_batch_end" in optionals else None
            if on_exit:
                on_exit(batch_result["context"])
            print("Cleaning up actors...")
            context = batch_result["context"]
            stop_actors(context.actor_map)
            stop_actors(context.sensor_map)

        return inner

    return decorator


def segment(
    frame_duration: int, segment_base_folder: Optional[FlexiblePath] = None
) -> Callable[[Segment[TContext, Any]], DecoratedSegment[TContext]]:
    """Decorator for a segment function. A segment function is a function that
    creates a small segment of data, e.g. 60 seconds of driving. It is provided the
    context by the batch, but the only thing it should change should be the things
    like position of the ego vehicle, the weather or spawn actors. A segment should
    not change the map or the world settings.

    Args:
        frame_duration (int): The number of frames to run the segment for.
        This will be dependent on the frame rate.


    Returns:
        Callable[[Segment[TContext, TSensorDataMap]], DecoratedSegment[TContext]]: The decorator.
    """

    # This function is used to be able to group the data generation into segments
    # Each segment is a snippet of any frame length. It is provided the client
    # by the batch, but can itself also change the world settings, like the map
    # and weather.
    # Each segment has:
    # - a configure stage (set up actors, sensors, etc.)
    # - a run stage with tasks (running the game loop with its tasks)
    # - a cleanup stage (clean up actors, sensors, etc.)
    def decorator(
        func: Segment[TContext, TSensorDataMap]
    ) -> DecoratedSegment[TContext]:
        @wraps(func)
        def inner(context: TContext, batch_base_path: Path) -> None:
            segment_result = func(context)
            print("Starting segment")
            tasks = segment_result["tasks"]
            optionals = segment_result["options"]
            segment_path = _flexible_path_to_path(segment_base_folder)
            save_items_base_path = batch_base_path / segment_path
            cleanup_actors = (
                optionals["cleanup_actors"] if "cleanup_actors" in optionals else False
            )
            on_finish_save_files = (
                optionals["on_finish_save_files"]
                if "on_finish_save_files" in optionals
                else None
            )

            on_exit = (
                optionals["on_segment_end"] if "on_segment_end" in optionals else None
            )

            def on_end(ctx: TContext, save_files_base_path: Path):
                if on_finish_save_files is not None:
                    save_items = on_finish_save_files(ctx)
                    save_items_to_file(save_files_base_path, save_items)
                    # To make sure the files are written to disk before next step
                if on_exit is not None:
                    on_exit(ctx, save_files_base_path)

            game_loop_segment(
                context=context,
                tasks=tasks,
                on_finished=on_end,
                max_frames=frame_duration,
                save_files_base_path=save_items_base_path,
                cleanup_actors=cleanup_actors,
            )

        return inner

    return decorator


def create_segment(
    frame_duration: Optional[int],
    segment_base_folder: Optional[FlexiblePath],
    segment_config: FullSegmentConfig[TSettings, TContext, TSensorDataMap],
) -> FullSegment:

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
    segments: List[FullSegment],
    settings: TSettings,
    after_segment_end: Optional[Callable[[TSettings], None]] = None,
):
    for segment in segments:
        segment(settings)
        if after_segment_end is not None:
            after_segment_end(settings)
