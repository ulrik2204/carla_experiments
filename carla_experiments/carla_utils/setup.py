import sys
import time
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    TypedDict,
    TypeVar,
    cast,
)

import carla

from carla_experiments.carla_utils.spawn import spawn_ego_vehicle, spawn_sensor
from carla_experiments.carla_utils.types_carla_utils import (
    CarlaTask,
    SensorBlueprint,
    SensorBlueprintCollection,
)

TActors = TypeVar("TActors")

TSensorData = TypeVar("TSensorData")


class Subclass(TypedDict, total=False):
    optional1: str
    optional2: str


class Super(TypedDict):
    required1: str
    required2: str
    optionals: Subclass


class SensorConfig(TypedDict, Generic[TSensorData]):
    blueprint: SensorBlueprint[TSensorData]
    location: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    attributes: Mapping[str, Any]


AvailableMaps = Literal[
    "Town01",
    "Town02",
    "Town03",
    "Town04",
    "Town05",
    "Town06",
    "Town07",
    "Town10",
    "Town11",
    "Town12",
]


class Sen(Generic[TSensorData], TypedDict):
    ...


TSensorsMap = TypeVar("TSensorsMap", bound=Mapping[str, Any])
TSensorDataMap = TypeVar("TSensorDataMap", bound=Mapping[str, Any])


def initialize_carla():
    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True  # Enables synchronous mode
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    return client, world


# def parse_carla_image(image: carla.Image) -> np.ndarray:
#     buffer_arr = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
#     buffer_arr_cpy = copy.deepcopy(buffer_arr)
#     return np.reshape(buffer_arr_cpy, (image.height, image.width, 4))


# def parse_radar(radar_data: carla.RadarMeasurement) -> np.ndarray:
#     # [depth, azimuth, altitute, velocity]
#     points = np.frombuffer(radar_data.raw_data, dtype=np.dtype("f4"))
#     points_cpy = copy.deepcopy(points)
#     points_shaped = np.reshape(points_cpy, (int(points_cpy.shape[0] / 4), 4))
#     return np.flip(points_shaped, 1)


# def parse_collision_event(collision_event: carla.CollisionEvent):
#     pass


# def parse_gnss_measurement(gnss_data: carla.GnssMeasurement) -> np.ndarray:
#     return np.array(
#         [gnss_data.latitude, gnss_data.longitude, gnss_data.altitude], dtype=np.float64
#     )


# def parse_imu_measurement(imu_data: carla.IMUMeasurement) -> np.ndarray:
#     return np.array(
#         [
#             imu_data.accelerometer.x,
#             imu_data.accelerometer.y,
#             imu_data.accelerometer.z,
#             imu_data.gyroscope.x,
#             imu_data.gyroscope.y,
#             imu_data.gyroscope.z,
#             imu_data.compass,
#         ],
#         dtype=np.float64,
#     )


# def parse_lidar(data: carla.LidarMeasurement):
#     buffer_points = np.frombuffer(data.raw_data, dtype=np.dtype("f4"))
#     buffer_points_cpy = copy.deepcopy(buffer_points)
#     return np.reshape(buffer_points_cpy, (int(buffer_points_cpy.shape[0] / 4), 4))


# def parse_lane_invasion(lane_invasion_event: carla.LaneInvasionEvent):
#     return np.array(
#         [
#             str(lane_marking.type)
#             for lane_marking in lane_invasion_event.crossed_lane_markings
#         ]
#     )


# def parse_obstacle(obstacle_event: carla.ObstacleDetectionEvent):
#     # TODO: Is this the correct way to handle obstacle detection event?
#     actor_id = obstacle_event.actor.id
#     other_actor_id = obstacle_event.other_actor.id
#     distance = obstacle_event.distance

#     return np.array([actor_id, other_actor_id, distance])


# def parse_speedometer(data: float):
#     # TODO: Implement
#     raise NotImplementedError()


# def parse_opendrive_map(data: str):
#     # TODO: Implement
#     raise NotImplementedError()


def create_camera_blueprint_modifier(
    width: Optional[int] = None, height: Optional[int] = None, fov: Optional[int] = None
):
    def inner(blueprint: carla.ActorBlueprint):
        blueprint.set_attribute("image_size_x", str(width or 1280))
        blueprint.set_attribute("image_size_y", str(height or 720))
        blueprint.set_attribute("fov", str(fov or 105))
        blueprint.set_attribute("lens_circle_multiplier", str(3.0))
        blueprint.set_attribute("lens_circle_falloff", str(3.0))
        blueprint.set_attribute("chromatic_aberration_intensity", str(0.5))
        blueprint.set_attribute("chromatic_aberration_offset", str(0))
        return blueprint

    return inner


def modify_lidar_blueprint(blueprint: carla.ActorBlueprint):
    blueprint.set_attribute("range", str(85))
    blueprint.set_attribute("rotation_frequency", str(10))
    blueprint.set_attribute("channels", str(64))
    blueprint.set_attribute("upper_fov", str(10))
    blueprint.set_attribute("lower_fov", str(-30))
    blueprint.set_attribute("points_per_second", str(600000))
    blueprint.set_attribute("atmosphere_attenuation_rate", str(0.004))
    blueprint.set_attribute("dropoff_general_rate", str(0.45))
    blueprint.set_attribute("dropoff_intensity_limit", str(0.8))
    blueprint.set_attribute("dropoff_zero_intensity", str(0.4))
    return blueprint


def create_radar_blueprint_modifier(fov: Optional[int] = None):
    def inner(blueprint: carla.ActorBlueprint):
        blueprint.set_attribute("horizontal_fov", str(fov or 45))  # degrees
        blueprint.set_attribute("vertical_fov", str(fov or 45))  # degrees
        blueprint.set_attribute("points_per_second", "1500")
        blueprint.set_attribute("range", "100")  # meters
        return blueprint

    return inner


def modify_gnss_blueprint(blueprint: carla.ActorBlueprint):
    blueprint.set_attribute("noise_alt_stddev", str(0.000005))
    blueprint.set_attribute("noise_lat_stddev", str(0.000005))
    blueprint.set_attribute("noise_lon_stddev", str(0.000005))
    blueprint.set_attribute("noise_alt_bias", str(0.0))
    blueprint.set_attribute("noise_lat_bias", str(0.0))
    blueprint.set_attribute("noise_lon_bias", str(0.0))
    return blueprint


def modify_imu_blueprint(blueprint: carla.ActorBlueprint):
    blueprint.set_attribute("noise_accel_stddev_x", str(0.001))
    blueprint.set_attribute("noise_accel_stddev_y", str(0.001))
    blueprint.set_attribute("noise_accel_stddev_z", str(0.015))
    blueprint.set_attribute("noise_gyro_stddev_x", str(0.001))
    blueprint.set_attribute("noise_gyro_stddev_y", str(0.001))
    blueprint.set_attribute("noise_gyro_stddev_z", str(0.001))
    return blueprint


def _handle_sensor_setup(
    world: carla.World,
    ego_vehicle: carla.Vehicle,
    sensor_id: str,
    sensor_config: SensorConfig,
    sensor_data_queue: Queue,
) -> carla.Sensor:
    sensor_blueprint = sensor_config["blueprint"]
    attributes = sensor_config["attributes"]
    # TODO: Handle Speedometer and Opendrive Map
    # if sensor_type.startswith("sensor.opendrive_map"):
    #     # The HDMap pseudo sensor is created directly here
    #     sensor = OpenDriveMapReader(vehicle, sensor_config["reading_frequency"])
    # elif sensor_type.startswith("sensor.speedometer"):
    #     delta_time = CarlaDataProvider.get_world().get_settings().fixed_delta_seconds
    #     frame_rate = 1 / delta_time
    #     sensor = SpeedometerReader(vehicle, frame_rate)
    # # These are the sensors spawned on the carla world
    # else:
    modify_camera_blueprint = create_camera_blueprint_modifier(
        attributes.get("width"), attributes.get("height"), attributes.get("fov")
    )
    modify_radar_blueprint = create_radar_blueprint_modifier(attributes.get("fov"))

    modify_blueprint_fn_map: Dict[
        SensorBlueprint[Any],
        Optional[Callable[[carla.ActorBlueprint], carla.ActorBlueprint]],
    ] = {
        SensorBlueprintCollection.CAMERA_RGB: modify_camera_blueprint,
        SensorBlueprintCollection.CAMERA_DEPTH: modify_camera_blueprint,
        SensorBlueprintCollection.CAMERA_DVS: modify_camera_blueprint,
        SensorBlueprintCollection.CAMERA_SEMANTIC_SEGMENTATION: modify_camera_blueprint,
        SensorBlueprintCollection.COLLISION: None,
        SensorBlueprintCollection.GNSS: modify_gnss_blueprint,
        SensorBlueprintCollection.IMU: modify_imu_blueprint,
        SensorBlueprintCollection.LANE_INVASION: None,
        SensorBlueprintCollection.OBSTACLE: None,
        SensorBlueprintCollection.LIDAR_RANGE: modify_lidar_blueprint,
        SensorBlueprintCollection.LIDAR_SEMANTIC_SEGMENTATION: modify_lidar_blueprint,
        SensorBlueprintCollection.RADAR_RANGE: modify_radar_blueprint,
        SensorBlueprintCollection.SPEEDOMETER: None,  # TODO: Implement
        SensorBlueprintCollection.OPENDRIVE_MAP: None,  # TODO: Implement
    }
    if sensor_blueprint in modify_blueprint_fn_map:
        modify_blueprint_fn = modify_blueprint_fn_map[sensor_blueprint]
    else:
        raise ValueError(f"Unknown sensor blueprint {sensor_blueprint}")

    # sensor_data_parsing_map: Dict[SensorBlueprint[Any], Callable[[Any], Any]] = {
    #     SensorBlueprintCollection.CAMERA_RGB: parse_carla_image,
    #     SensorBlueprintCollection.CAMERA_DEPTH: parse_carla_image,
    #     SensorBlueprintCollection.CAMERA_DVS: parse_carla_image,
    #     SensorBlueprintCollection.CAMERA_SEMANTIC_SEGMENTATION: parse_carla_image,
    #     SensorBlueprintCollection.COLLISION: parse_collision_event,
    #     SensorBlueprintCollection.GNSS: parse_gnss_measurement,
    #     SensorBlueprintCollection.IMU: parse_imu_measurement,
    #     SensorBlueprintCollection.LANE_INVASION: parse_lane_invasion,
    #     SensorBlueprintCollection.OBSTACLE: parse_obstacle,
    #     SensorBlueprintCollection.LIDAR_RANGE: parse_lidar,
    #     SensorBlueprintCollection.LIDAR_SEMANTIC_SEGMENTATION: lidar_semantic_segmentation_callback,
    #     SensorBlueprintCollection.RADAR_RANGE: radar_range_callback,
    #     SensorBlueprintCollection.SPEEDOMETER: parse_speedometer,
    #     SensorBlueprintCollection.OPENDRIVE_MAP: parse_opendrive_map,
    # }
    # if sensor_blueprint in sensor_data_parsing_map:
    #     parse_fn = sensor_data_parsing_map[sensor_blueprint]
    # else:
    #     raise ValueError(f"Unknown sensor blueprint {sensor_blueprint}")

    def on_measurement_received(data: carla.SensorData):
        # parsed_data = parse_fn(data)
        # TODO: HERE
        # DO I NEED TO PARSE THE DATA INTO A NUMPY ARRAY, OR CAN IT STAY AS A SensorData OBJECT?
        sensor_data_queue.put((sensor_id, data))

    sensor = spawn_sensor(
        world,
        sensor_blueprint,
        sensor_config["location"],
        sensor_config["rotation"],
        ego_vehicle,
        modify_blueprint_fn=modify_blueprint_fn,
        on_measurement_received=on_measurement_received,
    )
    return sensor


@dataclass
class CarlaSimulationEnvironment(Generic[TActors, TSensorsMap, TSensorDataMap]):
    client: carla.Client
    world: carla.World
    traffic_manager: carla.TrafficManager
    ego_vehicle: carla.Vehicle
    sensor_map: TSensorsMap
    sensor_data_queue: Queue
    # TODO: Add actors
    # other_actors_map: TActors
    _sensor_data_dict: Optional[TSensorDataMap] = field(init=False, default=None)

    def pre_tick(self, queue_timeout: float = 10):
        try:
            data_dict = {}
            while len(data_dict.keys()) < len(self.sensor_map.keys()):
                # # Don't wait for the opendrive sensor
                # if self._opendrive_tag and self._opendrive_tag not in data_dict.keys() \
                #         and len(self._sensors_objects.keys()) == len(data_dict.keys()) + 1:
                #     break

                sensor_id, data = self.sensor_data_queue.get(True, queue_timeout)
                data_dict[sensor_id] = data

        except Empty:
            # TODO: Maybe not throw exception?
            raise Exception("A sensor took too long to send their data")

        self._sensor_data_dict = cast(TSensorDataMap, data_dict)

    def tick(self):
        self.world.tick()

    def get_sensor_data(self) -> TSensorDataMap:
        if self._sensor_data_dict is None:
            raise Exception("Must call pre_tick before calling get_sensor_data")
        return self._sensor_data_dict


def initialize_carla_with_vehicle_and_sensors(
    map: AvailableMaps,
    sensor_config: Mapping[str, SensorConfig],
    frame_rate: int = 20,
    ego_vehicle_autopilot: bool = True,
    ego_vehicle_blueprint: str = "vehicle.tesla.model3",
    ego_vehicle_spawn_point: Optional[
        Callable[[List[carla.Transform]], carla.Transform]
    ] = None,
):
    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)
    client.load_world(map)
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True  # Enables synchronous mode
    settings.fixed_delta_seconds = 1.0 / frame_rate
    world.apply_settings(settings)

    ego_vehicle = spawn_ego_vehicle(
        world,
        blueprint=ego_vehicle_blueprint,
        autopilot=ego_vehicle_autopilot,
        choose_spawn_point=ego_vehicle_spawn_point,
    )
    # time.sleep(1)
    # ego_vehicle.set_autopilot(True)
    sensor_map: Mapping[str, carla.Sensor] = {}
    sensor_data_queue = Queue()

    for sensor_id, config in sensor_config.items():
        sensor = _handle_sensor_setup(
            world, ego_vehicle, sensor_id, config, sensor_data_queue
        )
        sensor_map[sensor_id] = sensor

    # Do a single tick to spawn the sensors
    world.tick()
    return CarlaSimulationEnvironment(
        client=client,
        world=world,
        ego_vehicle=ego_vehicle,
        sensor_map=sensor_map,
        sensor_data_queue=sensor_data_queue,
        traffic_manager=client.get_trafficmanager(),
    )


TEnv = TypeVar("TEnv", bound=CarlaSimulationEnvironment[Any, Any, Any])


def game_loop_environment(
    environment: TEnv,
    tasks: List[Callable[[TEnv], None]],
):
    while True:
        try:  # in case of a crash, try to recover and continue
            environment.pre_tick()
            for task in tasks:
                task(environment)  # type: ignore
            time.sleep(0.01)
            environment.tick()
        except (KeyboardInterrupt, Exception):
            print("Exiting...")
            # TODO: handle actors
            # for actor in actors.values():  # type: ignore
            #     actor = cast(carla.Actor, actor)
            #     actor.destroy()
            for sensor in environment.sensor_map.values():
                sensor.stop()
                sensor.destroy()
            sys.exit()


def game_loop(
    world: carla.World,
    tasks: List[CarlaTask[TActors]],
    actors: TActors,
):
    while True:
        try:  # in case of a crash, try to recover and continue
            for task in tasks:
                task(world, actors)  # type: ignore
            time.sleep(0.01)
            world.tick()
        except (KeyboardInterrupt, Exception):
            print("Exiting...")
            for actor in actors.values():  # type: ignore
                actor = cast(carla.Actor, actor)
                actor.destroy()  # type: ignore
            # actors = {}
            sys.exit()
