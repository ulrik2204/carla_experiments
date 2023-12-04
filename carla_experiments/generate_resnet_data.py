import json
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import carla

from carla_experiments.carla_utils.sensors import spawn_sensor
from carla_experiments.carla_utils.setup import game_loop, initialize_carla
from carla_experiments.carla_utils.types_carla_utils import SensorBlueprintCollection
from carla_experiments.tutorial_generate_data import spawn_ego_vehicle


class ResnetActors(TypedDict):
    ego_vehicle: carla.Vehicle
    camera: carla.Sensor


def follow_camera_task(world: carla.World, actors: ResnetActors) -> ResnetActors:
    ego_vehicle = actors["ego_vehicle"]
    spectator = world.get_spectator()
    spectator_transform = carla.Transform(
        ego_vehicle.get_transform().transform(carla.Location(z=4)),
        carla.Rotation(pitch=-15),
    )
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

    spectator = world.get_spectator()
    spectator_transform = carla.Transform(
        ego_vehicle.get_transform().transform(carla.Location(z=4)),
        carla.Rotation(pitch=-15),
    )
    spectator.set_transform(spectator_transform)

    return ResnetActors(ego_vehicle=ego_vehicle, camera=rgb_cam)


def main():
    _, world = initialize_carla()
    tasks = []
    actors = initialize_actors(world)
    game_loop(world, tasks, actors)


if __name__ == "__main__":
    main()
