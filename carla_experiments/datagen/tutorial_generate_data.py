from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, TypeVar

import carla

from carla_experiments.carla_utils.constants import SensorBlueprints
from carla_experiments.carla_utils.spawn import spawn_ego_vehicle, spawn_sensor

TSensorData = TypeVar("TSensorData")


@dataclass
class CarlaContext:
    sensor_list: List[carla.Sensor]
    spectator: carla.Actor


def initialize_carla():
    client = carla.Client("localhost", 2000)
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True  # Enables synchronous mode
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    return client, world


def main():
    _, world = initialize_carla()
    timestampstr = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sensor_list: List[carla.Actor] = []
    spectator = world.get_spectator()
    # world_snapshot =
    # world.wait_for_tick()
    ego_vehicle = spawn_ego_vehicle(world)
    ego_vehicle.set_autopilot(True)

    spectator = world.get_spectator()
    vehicle_transform = ego_vehicle.get_transform()
    spectator_transform = carla.Transform(
        vehicle_transform.transform(carla.Location(x=-8, z=4)),
        carla.Rotation(pitch=-15),
    )
    spectator.set_transform(spectator_transform)

    def save_image_to_disk(image: carla.Image):
        image.save_to_disk(f"output/{timestampstr}/{image.frame:6d}.jpg")

    def set_camera_attributes(cam_bp: carla.ActorBlueprint):
        cam_bp.set_attribute("image_size_x", str(1920))
        cam_bp.set_attribute("image_size_y", str(1080))
        cam_bp.set_attribute("fov", str(105))
        return cam_bp

    rgb_cam = spawn_sensor(
        world,
        SensorBlueprints.CAMERA_RGB,
        location=(2, 0, 1),
        rotation=(0, 0, 0),
        attach_to=ego_vehicle,
        modify_blueprint_fn=set_camera_attributes,
        on_measurement_received=save_image_to_disk,
    )
    sensor_list.append(rgb_cam)
    time.sleep(1)

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
            # Update the spectator's transform to follow the vehicle
            vehicle_transform = ego_vehicle.get_transform()
            spectator_transform.location = vehicle_transform.location + carla.Location(
                z=4
            )
            spectator_transform.rotation.yaw = vehicle_transform.rotation.yaw
            spectator.set_transform(spectator_transform)
            time.sleep(0.01)

            world.tick()
        except KeyboardInterrupt:
            [sensor.destroy() for sensor in sensor_list]
            ego_vehicle.destroy()
            sys.exit()


if __name__ == "__main__":
    main()
