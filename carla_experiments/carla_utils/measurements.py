import copy
from typing import Tuple, TypedDict

import carla
import numpy as np


def calculate_vehicle_speed(vehicle: carla.Vehicle) -> float:
    velocity = vehicle.get_velocity()
    transform = vehicle.get_transform()

    vel_np = np.array([velocity.x, velocity.y, velocity.z])
    pitch = np.deg2rad(transform.rotation.pitch)
    yaw = np.deg2rad(transform.rotation.yaw)
    orientation = np.array(
        [
            np.cos(pitch) * np.cos(yaw),
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
        ]
    )
    return float(np.dot(vel_np, orientation))


def parse_radar_data(radar_data: carla.RadarMeasurement) -> np.ndarray:
    points = np.frombuffer(radar_data.raw_data, dtype=np.dtype("f4"))
    points_cpy = copy.deepcopy(points)
    points_cpy = np.reshape(points_cpy, (int(points_cpy.shape[0] / 4), 4))
    points_cpy = np.flip(points_cpy, 1)
    return points_cpy


def parse_image_data(image: carla.Image) -> np.ndarray:
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array_cpy = copy.deepcopy(array)
    array_cpy = np.reshape(array_cpy, (image.height, image.width, 4))
    return array_cpy


def parse_lidar_data(lidar_data: carla.LidarMeasurement) -> np.ndarray:
    points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype("f4"))
    points_cpy = copy.deepcopy(points)
    points_cpy = np.reshape(points_cpy, (int(points_cpy.shape[0] / 4), 4))
    return points_cpy


class GNSSData(TypedDict):
    latitude: float
    longitude: float
    altitude: float


def parse_gnss_data(gnss_data: carla.GnssMeasurement) -> GNSSData:
    return {
        "latitude": gnss_data.latitude,
        "longitude": gnss_data.longitude,
        "altitude": gnss_data.altitude,
    }


class IMUData(TypedDict):
    accelerometer: Tuple[float, float, float]
    gyroscope: Tuple[float, float, float]
    compass: float


def parse_imu_data(imu_data: carla.IMUMeasurement) -> IMUData:
    return {
        "accelerometer": (
            imu_data.accelerometer.x,
            imu_data.accelerometer.y,
            imu_data.accelerometer.z,
        ),
        "gyroscope": (imu_data.gyroscope.x, imu_data.gyroscope.y, imu_data.gyroscope.z),
        "compass": imu_data.compass,
    }


class WaypointData(TypedDict):
    location: Tuple[float, float, float]
    rotation: Tuple[float, float, float]


def parse_waypoint(waypoint: carla.Waypoint) -> WaypointData:
    return {
        "location": (
            waypoint.transform.location.x,
            waypoint.transform.location.y,
            waypoint.transform.location.z,
        ),
        "rotation": (
            waypoint.transform.rotation.pitch,
            waypoint.transform.rotation.yaw,
            waypoint.transform.rotation.roll,
        ),
    }
