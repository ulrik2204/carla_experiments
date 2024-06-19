from typing import Generic, TypeVar

import carla

from openpilot_exploration.carla_utils.types_carla_utils import (
    Constant,
    SensorBlueprint,
)

TSensorData = TypeVar("TSensorData")


class SensorBlueprints(Constant, Generic[TSensorData]):
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


class AttributeDefaults(Constant):
    CAMERA = {
        "image_size_x": "512",
        "image_size_y": "256",
        "fov": "105",
        "lens_circle_multiplier": "3.0",
        "lens_circle_falloff": "3.0",
        "chromatic_aberration_intensity": "0.5",
        "chromatic_aberration_offset": "0",
    }
    LIDAR = {
        "range": "85",
        "rotation_frequency": "10",
        "channels": "64",
        "upper_fov": "10",
        "lower_fov": "-30",
        "points_per_second": "600000",
        "atmosphere_attenuation_rate": "0.004",
        "dropoff_general_rate": "0.45",
        "dropoff_intensity_limit": "0.8",
        "dropoff_zero_intensity": "0.4",
    }
    RADAR = {
        "horizontal_fov": "45",
        "vertical_fov": "45",
        "points_per_second": "1500",
        "range": "100",
    }
    IMU = {
        "noise_accel_stddev_x": "0.001",
        "noise_accel_stddev_y": "0.001",
        "noise_accel_stddev_z": "0.015",
        "noise_gyro_stddev_x": "0.001",
        "noise_gyro_stddev_y": "0.001",
        "noise_gyro_stddev_z": "0.001",
    }
    GNSS = {
        "noise_alt_stddev": "0.000005",
        "noise_lat_stddev": "0.000005",
        "noise_lon_stddev": "0.000005",
        "noise_alt_bias": "0.0",
        "noise_lat_bias": "0.0",
        "noise_lon_bias": "0.0",
    }
