import copy
import time
from typing import TypedDict

import carla
import numpy as np
import torch
from PIL import Image

from carla_experiments.carla_utils.setup import game_loop_environment, initialize_carla
from carla_experiments.carla_utils.spawn import spawn_ego_vehicle, spawn_single_sensor
from carla_experiments.carla_utils.types_carla_utils import SensorBlueprints
from carla_experiments.datasets.simple_dataset import get_simple_val_test_transforms
from carla_experiments.models.simple import SimpleLineFollowingB0
from carla_experiments.train.training_utils import load_state_dict


class SimpleEvalActors(TypedDict):
    ego_vehicle: carla.Vehicle
    camera: carla.Sensor


start_time = time.time()


def load_line_following_model():
    model = SimpleLineFollowingB0(3, True)
    model, *_ = load_state_dict(model, None, "./.weights/agentB0/091148-loss0.0007.pt")
    return model


line_following_model = load_line_following_model()


def predict_vehicle_controls(image):
    with torch.no_grad():
        output = line_following_model(image)
        output = output.squeeze(0)
        steer = float(output[0].item())
        throttle = float(output[1].item())
        brake = float(output[2].item())
        return steer, throttle, 0.0 if brake < 0.5 else brake


def get_ego_vehicle(actors: SimpleEvalActors) -> carla.Vehicle:
    return actors["ego_vehicle"]


def check_stop_time_task(world: carla.World, actors: SimpleEvalActors) -> None:
    global start_time
    max_time_seconds = 60 * 60 * 3  # 3 hours
    if time.time() - start_time > max_time_seconds:
        print("Max time reached, exiting")
        raise KeyboardInterrupt()


def follow_camera_task(world: carla.World, actors: SimpleEvalActors) -> None:
    ego_vehicle = get_ego_vehicle(actors)
    vehicle_transform = ego_vehicle.get_transform()
    spectator = world.get_spectator()
    spectator_transform = spectator.get_transform()
    spectator_transform.location = vehicle_transform.location + carla.Location(z=4)
    spectator_transform.rotation.yaw = vehicle_transform.rotation.yaw
    spectator.set_transform(spectator_transform)


def setup_ego_vehicle(world: carla.World) -> carla.Vehicle:
    ego_vehicle = spawn_ego_vehicle(world)
    # Sleep before setting autopilot is important because of timing issues.
    time.sleep(1)
    return ego_vehicle


def setup_camera(world: carla.World, ego_vehicle: carla.Vehicle) -> carla.Sensor:
    def set_camera_attributes(cam_bp: carla.ActorBlueprint):
        cam_bp.set_attribute("image_size_x", str(1920))
        cam_bp.set_attribute("image_size_y", str(1080))
        cam_bp.set_attribute("fov", str(105))
        return cam_bp

    def control_vehicle_from_image(image: carla.Image):
        # This should instead put the image into a queue or dict to gather data for the
        # model to control the vehicle with multiple sensors
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = copy.deepcopy(array)
        array = np.reshape(array, (image.height, image.width, 4))
        pil_image = Image.fromarray(array[:, :, :3])
        transforms = get_simple_val_test_transforms()
        transformed_image = transforms(pil_image).unsqueeze(0)  # type: ignore
        steer, throttle, brake = predict_vehicle_controls(transformed_image)
        if image.frame % 100 == 0:
            print(
                "Frame: ",
                image.frame,
                "Steer:",
                steer,
                "Throttle:",
                throttle,
                "Brake:",
                brake,
            )
            # print("Image: ", array)
        control = carla.VehicleControl(throttle, steer, brake)
        ego_vehicle.apply_control(control)

    rgb_cam = spawn_single_sensor(
        world,
        SensorBlueprints.CAMERA_RGB,
        location=(2, 0, 1),
        rotation=(0, 0, 0),
        attach_to=ego_vehicle,
        modify_blueprint_fn=set_camera_attributes,
        on_sensor_data_received=control_vehicle_from_image,
    )
    time.sleep(0.1)
    return rgb_cam


def initialize_actors(world: carla.World) -> SimpleEvalActors:
    ego_vehicle = setup_ego_vehicle(world)
    camera = setup_camera(world, ego_vehicle)

    world.tick()  # Tick once to get the first image before the loop
    # It is important to only tick once during initialization
    time.sleep(1)
    return SimpleEvalActors(ego_vehicle=ego_vehicle, camera=camera)


def main():
    print("Initializing Carla...")
    _, world = initialize_carla()
    print("Carla initialized")
    actors = initialize_actors(world)
    tasks = [follow_camera_task, check_stop_time_task]
    game_loop_environment(world, tasks, actors)


if __name__ == "__main__":
    main()
