from dataclasses import dataclass
from typing import Any, Generic, Mapping, Protocol, Type, TypeVar

import carla


class Constant:
    def __new__(cls):
        raise TypeError("Cannot instantiate constant class")


TSensorData = TypeVar("TSensorData")

# It is suppsed to have to be a Dict[str, carla.Actor], but it does not support extensions of carla.Actor
TActors = TypeVar("TActors", bound=Mapping[str, Any])


class CarlaTask(Protocol, Generic[TActors]):
    def __call__(self, world: carla.World, actors: TActors) -> None:
        ...


@dataclass
class SensorBlueprint(Generic[TSensorData]):
    name: str
    sensor_data_type: Type[TSensorData]


class SensorBlueprintCollection(Constant, Generic[TSensorData]):
    CAMERA_RGB = SensorBlueprint("sensor.camera.rgb", carla.Image)
    CAMERA_DEPTH = SensorBlueprint("sensor.camera.depth", carla.Image)
    CAMERA_SEMANTIC_SEGMENTATION = SensorBlueprint(
        "sensor.camera.semantic_segmentation", carla.Image
    )
    CAMERA_DVS = SensorBlueprint("sensor.camera.dvs", carla.Image)
    LIDAR_RANGE = SensorBlueprint("sensor.lidar.ray_cast", carla.LidarMeasurement)
    LIDAR_SEMANTIC_SEGMENTATION = SensorBlueprint(
        "sensor.lidar.semantic_segmentation", carla.LidarMeasurement
    )
    RADAR_RANGE = SensorBlueprint("sensor.other.radar", carla.RadarMeasurement)
    GNSS = SensorBlueprint("sensor.other.gnss", carla.GnssMeasurement)
    IMU = SensorBlueprint("sensor.other.imu", carla.IMUMeasurement)
    COLLISION = SensorBlueprint("sensor.other.collision", carla.CollisionEvent)
    LANE_INVASION = SensorBlueprint(
        "sensor.other.lane_invasion", carla.LaneInvasionEvent
    )
    OBSTACLE = SensorBlueprint("sensor.other.obstacle", carla.ObstacleDetectionEvent)
