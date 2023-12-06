import json
import time
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import carla

from carla_experiments.carla_utils.setup import game_loop, initialize_carla
from carla_experiments.carla_utils.spawn import spawn_ego_vehicle, spawn_sensor
from carla_experiments.carla_utils.types_carla_utils import SensorBlueprintCollection


class ResnetActors(TypedDict):
    ego_vehicle: carla.Vehicle
    camera: carla.Sensor


start_time = time.time()


def get_ego_vehicle(actors: ResnetActors) -> carla.Vehicle:
    return actors["ego_vehicle"]


def check_stop_time_task(world: carla.World, actors: ResnetActors) -> None:
    global start_time
    max_time_seconds = 60 * 60 * 3  # 3 hours
    if time.time() - start_time > max_time_seconds:
        print("Max time reached, exiting")
        raise KeyboardInterrupt()


def follow_camera_task(world: carla.World, actors: ResnetActors) -> None:
    ego_vehicle = get_ego_vehicle(actors)
    vehicle_transform = ego_vehicle.get_transform()
    spectator = world.get_spectator()
    spectator_transform = spectator.get_transform()
    spectator_transform.location = vehicle_transform.location + carla.Location(z=4)
    spectator_transform.rotation.yaw = vehicle_transform.rotation.yaw
    spectator.set_transform(spectator_transform)


def setup_ego_vehicle(world: carla.World) -> carla.Vehicle:
    ego_vehicle = spawn_ego_vehicle(world)
    time.sleep(
        1
    )  # Sleep before setting autopilot is important because of timing issues.
    ego_vehicle.set_autopilot(True)
    return ego_vehicle


def setup_camera(
    world: carla.World, ego_vehicle: carla.Vehicle, folder_base_path: Path
) -> carla.Sensor:
    folder_base_path.mkdir(parents=True, exist_ok=True)
    controls_base_path = folder_base_path / "controls"
    controls_base_path.mkdir(parents=True, exist_ok=True)
    images_base_path = folder_base_path / "images"
    images_base_path.mkdir(parents=True, exist_ok=True)

    def save_image_and_controls(image: carla.Image):
        control = ego_vehicle.get_control()
        waypoint = world.get_map().get_waypoint(ego_vehicle.get_location())
        state = {
            "steer": control.steer,
            "throttle": control.throttle,
            "brake": control.brake,
            "waypoint": waypoint,
        }
        with open(controls_base_path / f"{image.frame:6d}.json", "w") as f:
            json.dump(state, f)
        image.save_to_disk((images_base_path / f"{image.frame:6d}.jpg").as_posix())

    def set_camera_attributes(cam_bp: carla.ActorBlueprint):
        cam_bp.set_attribute("image_size_x", str(1920))
        cam_bp.set_attribute("image_size_y", str(1080))
        cam_bp.set_attribute("fov", str(105))
        return cam_bp

    rgb_cam = spawn_sensor(
        world,
        SensorBlueprintCollection.CAMERA_RGB,
        location=(2, 0, 1),
        rotation=(0, 0, 0),
        attach_to=ego_vehicle,
        modify_blueprint_fn=set_camera_attributes,
        on_measurement_received=save_image_and_controls,
    )
    time.sleep(0.1)
    return rgb_cam


def initialize_actors(world: carla.World) -> ResnetActors:
    ego_vehicle = setup_ego_vehicle(world)

    timestamp_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_base_path = Path(f"output/{timestamp_string}")
    rgb_cam = setup_camera(world, ego_vehicle, folder_base_path)
    world.tick()  # Tick once to get the first image before the loop
    # It is important to only tick once during initialization
    time.sleep(1)
    return ResnetActors(ego_vehicle=ego_vehicle, camera=rgb_cam)


def main():
    _, world = initialize_carla()
    actors = initialize_actors(world)
    tasks = [follow_camera_task, check_stop_time_task]
    game_loop(world, tasks, actors)


if __name__ == "__main__":
    main()
