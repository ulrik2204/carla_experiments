from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Generic, List, Tuple, Type, TypeVar, cast

import carla


class Globals:
    sensor_list: List[carla.Sensor] = []


global_vars = Globals()


class Constant:
    def __new__(cls):
        raise TypeError("Cannot instantiate constant class")


TSensorData = TypeVar("TSensorData")


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


def spawn_sensor(
    world: carla.World,
    blueprint: SensorBlueprint[TSensorData],
    location: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    attach_to: carla.Actor | None = None,
    attachment_type: carla.AttachmentType = carla.AttachmentType.Rigid,
    modify_blueprint_fn: Callable[
        [carla.ActorBlueprint], carla.ActorBlueprint
    ] = lambda x: x,
    on_measurement_received: Callable[[TSensorData], None] = lambda _: None,
    global_sensor_list: List[carla.Sensor] = global_vars.sensor_list,
) -> carla.Sensor:
    sensor_blueprint = world.get_blueprint_library().find(blueprint.name)
    sensor_blueprint = modify_blueprint_fn(sensor_blueprint)
    sensor_location = carla.Location(*location)
    sensor_rotation = carla.Rotation(*rotation)

    sensor_transform = carla.Transform(sensor_location, sensor_rotation)
    sensor_object = cast(
        carla.Sensor,
        world.spawn_actor(
            sensor_blueprint,
            sensor_transform,
            attach_to,
            attachment_type,
        ),
    )

    sensor_object.listen(on_measurement_received)
    global_sensor_list.append(sensor_object)

    return sensor_object


def spawn_ego_vehicle(world: carla.World) -> carla.Vehicle:
    ego_bp = world.get_blueprint_library().find("vehicle.tesla.model3")
    ego_bp.set_attribute("role_name", "ego")
    print("\nEgo role_name is set")
    ego_color = random.choice(ego_bp.get_attribute("color").recommended_values)
    ego_bp.set_attribute("color", ego_color)
    print("\nEgo color is set")

    spawn_points = world.get_map().get_spawn_points()
    number_of_spawn_points = len(spawn_points)

    if 0 < number_of_spawn_points:
        random.shuffle(spawn_points)
        ego_transform = spawn_points[0]
        ego_vehicle = cast(carla.Vehicle, world.spawn_actor(ego_bp, ego_transform))
        print("\nEgo is spawned")
    else:
        raise Exception("Could not find any spawn points")

    # --------------
    # Spectator on ego position
    # --------------
    spectator = world.get_spectator()
    # world_snapshot =
    world.wait_for_tick()
    spectator.set_transform(ego_vehicle.get_transform())
    return ego_vehicle


def main():
    client = carla.Client("localhost", 2000)
    world = client.get_world()
    timestampstr = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ego_vehicle = spawn_ego_vehicle(world)
    ego_vehicle.set_autopilot(True)

    def save_image_to_disk(image: carla.Image):
        image.save_to_disk(f"output/{timestampstr}/{image.frame:6d}.jpg")

    # --------------
    # Spawn attached RGB camera
    # --------------cam_bp = None
    # cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    # cam_bp.set_attribute("image_size_x", str(1920))
    # cam_bp.set_attribute("image_size_y", str(1080))
    # cam_bp.set_attribute("fov", str(105))
    # cam_location = carla.Location(2, 0, 1)
    # cam_rotation = carla.Rotation(0, 180, 0)
    # cam_transform = carla.Transform(cam_location, cam_rotation)
    # ego_cam = cast(
    #     carla.Sensor,
    #     world.spawn_actor(
    #         cam_bp,
    #         cam_transform,
    #         ego_vehicle,
    #         carla.AttachmentType.Rigid,
    #     ),
    # )
    # ego_cam.listen(something)

    def set_camera_attributes(cam_bp: carla.ActorBlueprint):
        cam_bp.set_attribute("image_size_x", str(1920))
        cam_bp.set_attribute("image_size_y", str(1080))
        cam_bp.set_attribute("fov", str(105))
        return cam_bp

    spawn_sensor(
        world,
        SensorBlueprintCollection.CAMERA_RGB,
        location=(2, 0, 1),
        rotation=(0, 0, 0),
        attach_to=ego_vehicle,
        modify_blueprint_fn=set_camera_attributes,
        on_measurement_received=save_image_to_disk,
    )
    # --------------
    # Add collision sensor to ego vehicle.
    # --------------

    # def col_callback(colli):
    #     print("Collision detected:\n" + str(colli) + "\n")

    # spawn_sensor(
    #     world,
    #     SensorBlueprintCollection.COLLISION,
    #     (0, 0, 0),
    #     (0, 0, 0),
    #     ego_vehicle,
    #     on_measurement_received=col_callback,
    # )

    # --------------
    # Add Lane invasion sensor to ego vehicle.
    # --------------
    # def lane_callback(lane):
    #     print("Lane invasion detected:\n" + str(lane) + "\n")

    # spawn_sensor(
    #     world,
    #     SensorBlueprintCollection.LANE_INVASION,
    #     (0, 0, 0),
    #     (0, 0, 0),
    #     ego_vehicle,
    #     on_measurement_received=lane_callback,
    # )

    # --------------
    # Add Obstacle sensor to ego vehicle.
    # --------------
    # def obs_callback(obs):
    #     print("Obstacle detected:\n" + str(obs) + "\n")

    # spawn_sensor(
    #     world,
    #     SensorBlueprintCollection.OBSTACLE,
    #     (0, 0, 0),
    #     (0, 0, 0),
    #     ego_vehicle,
    #     on_measurement_received=obs_callback,
    # )

    # --------------
    # Add GNSS sensor to ego vehicle.
    # --------------

    # def gnss_callback(gnss):
    #     print("GNSS measure:\n" + str(gnss) + "\n")

    # spawn_sensor(
    #     world,
    #     SensorBlueprintCollection.GNSS,
    #     (0, 0, 0),
    #     (0, 0, 0),
    #     ego_vehicle,
    #     on_measurement_received=gnss_callback,
    # )

    # --------------
    # Add IMU sensor to ego vehicle.
    # --------------

    # def imu_callback(imu):
    #     print("IMU measure:\n" + str(imu) + "\n")

    # spawn_sensor(
    #     world,
    #     SensorBlueprintCollection.IMU,
    #     (0, 0, 0),
    #     (0, 0, 0),
    #     ego_vehicle,
    #     on_measurement_received=imu_callback,
    # )

    while True:
        try:
            world.tick()
        except KeyboardInterrupt:
            [sensor.destroy() for sensor in global_vars.sensor_list]
            sys.exit()


if __name__ == "__main__":
    main()
