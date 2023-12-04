import json
import sys
import time
from datetime import datetime
from pathlib import Path

import carla

from carla_experiments.carla_utils.sensors import spawn_sensor
from carla_experiments.carla_utils.setup import initialize_carla
from carla_experiments.carla_utils.types_carla_utils import SensorBlueprintCollection
from carla_experiments.tutorial_generate_data import spawn_ego_vehicle

# class ResnetActors(TypedDict):
#     ego_vehicle: carla.Vehicle
#     camera: carla.Sensor

ResnetActors = list


start_time = time.time()


def get_ego_vehicle(actors: ResnetActors) -> carla.Vehicle:
    return actors[0]
    return actors["ego_vehicle"]


def check_stop_time_task(world: carla.World, actors: ResnetActors) -> ResnetActors:
    global start_time
    max_time_seconds = 60 * 60 * 3  # 3 hours
    if time.time() - start_time > max_time_seconds:
        print("Max time reached, exiting")
        raise KeyboardInterrupt()
    return actors


def follow_camera_task(world: carla.World, actors: ResnetActors) -> ResnetActors:
    ego_vehicle = get_ego_vehicle(actors)
    vehicle_transform = ego_vehicle.get_transform()
    spectator = world.get_spectator()
    spectator_transform = spectator.get_transform()
    spectator_transform.location = vehicle_transform.location + carla.Location(z=4)
    spectator_transform.rotation.yaw = vehicle_transform.rotation.yaw
    spectator.set_transform(spectator_transform)
    return actors


def initialize_actors(world: carla.World) -> ResnetActors:
    timestampstr = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ego_vehicle = spawn_ego_vehicle(world)
    ego_vehicle.set_autopilot(True)
    folder_base_path = Path(f"output/{timestampstr}")
    folder_base_path.mkdir(parents=True, exist_ok=True)
    controls_base_path = folder_base_path / "controls"
    controls_base_path.mkdir(parents=True, exist_ok=True)
    images_base_path = folder_base_path / "images"
    images_base_path.mkdir(parents=True, exist_ok=True)

    def save_image_and_controls(image: carla.Image):
        control = ego_vehicle.get_control()
        state = {
            "steer": control.steer,
            "throttle": control.throttle,
            "brake": control.brake,
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

    time.sleep(1)

    actors = [ego_vehicle, rgb_cam]
    return actors


# def main():
#     _, world = initialize_carla()
#     tasks = [follow_camera_task, check_stop_time_task]
#     actors = initialize_actors(world)
#     time.sleep(0.1)
#     game_loop(world, tasks, actors)  # type: ignore


def main():
    _, world = initialize_carla()
    tasks = [follow_camera_task, check_stop_time_task]
    actors = initialize_actors(world)

    # timestampstr = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # ego_vehicle = spawn_ego_vehicle(world)
    # ego_vehicle.set_autopilot(True)
    # folder_base_path = Path(f"output/{timestampstr}")
    # folder_base_path.mkdir(parents=True, exist_ok=True)
    # controls_base_path = folder_base_path / "controls"
    # controls_base_path.mkdir(parents=True, exist_ok=True)
    # images_base_path = folder_base_path / "images"
    # images_base_path.mkdir(parents=True, exist_ok=True)

    # def save_image_and_controls(image: carla.Image):
    #     control = ego_vehicle.get_control()
    #     state = {
    #         "steer": control.steer,
    #         "throttle": control.throttle,
    #         "brake": control.brake,
    #     }
    #     with open(controls_base_path / f"{image.frame:6d}.json", "w") as f:
    #         json.dump(state, f)
    #     image.save_to_disk((images_base_path / f"{image.frame:6d}.jpg").as_posix())

    # def set_camera_attributes(cam_bp: carla.ActorBlueprint):
    #     cam_bp.set_attribute("image_size_x", str(1920))
    #     cam_bp.set_attribute("image_size_y", str(1080))
    #     cam_bp.set_attribute("fov", str(105))
    #     return cam_bp

    # rgb_cam = spawn_sensor(
    #     world,
    #     SensorBlueprintCollection.CAMERA_RGB,
    #     location=(2, 0, 1),
    #     rotation=(0, 0, 0),
    #     attach_to=ego_vehicle,
    #     modify_blueprint_fn=set_camera_attributes,
    #     on_measurement_received=save_image_and_controls,
    # )
    # actors = [ego_vehicle, rgb_cam]
    # follow_camera_task(world, actors)

    time.sleep(0.1)
    # game_loop(world, tasks, actors)  # type: ignore

    while True:
        try:  # in case of a crash, try to recover and continue
            for task in tasks:
                task(world, actors)  # type: ignore
            time.sleep(0.01)
            world.tick()
        except (KeyboardInterrupt, Exception):
            print("Exiting...")
            for actor in actors:  # TODO: .values()
                actor.destroy()  # type: ignore
            sys.exit()


if __name__ == "__main__":
    main()
