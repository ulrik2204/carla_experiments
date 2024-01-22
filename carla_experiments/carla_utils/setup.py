import sys
import time
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Mapping, Optional, Type, TypeVar, cast

import carla

from carla_experiments.carla_utils.constants import AttributeDefaults, SensorBlueprints
from carla_experiments.carla_utils.spawn import spawn_sensor
from carla_experiments.carla_utils.types_carla_utils import (
    CarlaContext,
    SensorBlueprint,
    SensorConfig,
)

TSensorMap = TypeVar("TSensorMap", bound=Mapping[str, Any])
TSensorDataMap = TypeVar("TSensorDataMap", bound=Mapping[str, Any])
TActorMap = TypeVar("TActorMap", bound=Mapping[str, Any])


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


def setup_carla_client(map: str, frame_rate: int = 20):
    """Creates the CARLA client at given port and
    sets the world to synchronous mode with given frame rate.
    Also sets the timeout to 30 seconds.

    Args:
        map (str): The map to load. E.g. "Town01"
        frame_rate (int, optional): Frame rate to run the simulation. Defaults to 20.

    Returns:
        Client: The CARLA client object
    """
    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)
    client.load_world(map)
    world = client.get_world()
    client.get_trafficmanager().set_synchronous_mode(True)

    settings = world.get_settings()
    settings.synchronous_mode = True  # Enables synchronous mode
    settings.fixed_delta_seconds = 1.0 / frame_rate
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


TContext = TypeVar("TContext", bound=CarlaContext)


def _get_sensor_data_map(env: CarlaContext, queue_timeout: float = 10):
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


def game_loop(
    context: TContext,
    tasks: List[Callable[[TContext, TSensorDataMap], None]],
    on_exit: Optional[Callable[[TContext], None]] = None,
):
    while True:
        try:  # in case of a crash, try to recover and continue
            sensor_data_map = _get_sensor_data_map(context)
            for task in tasks:
                task(context, sensor_data_map)  # type: ignore
            time.sleep(0.01)
            context.client.get_world().tick()
        except (KeyboardInterrupt, Exception) as e:
            if not isinstance(e, KeyboardInterrupt):
                print(e)
            if on_exit is not None:
                on_exit(context)
            print("Exiting...")
            stop_actor(context.actor_map)
            stop_actor(context.sensor_map)
            sys.exit()


def stop_actor(actor):
    if isinstance(actor, carla.Actor):
        actor.destroy()
    elif isinstance(actor, carla.Sensor):
        actor.stop()
        actor.destroy()
    elif isinstance(actor, list) or isinstance(actor, tuple):
        for a in actor:
            stop_actor(a)
    elif isinstance(actor, dict):
        for a in actor.values():
            stop_actor(a)
    else:
        raise ValueError(f"Unsupposed actor map type {type(actor)}")
