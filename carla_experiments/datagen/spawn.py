from __future__ import annotations

import math
import queue
import random
from typing import List, cast

from carla import (
    Actor,
    ActorBlueprint,
    Client,
    Location,
    Rotation,
    Sensor,
    Transform,
    Vehicle,
    WalkerAIController,
)


def get_carla_client():
    client = Client("localhost", 2000)
    world = client.get_world()

    # Set up the simulator in synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True  # Enables synchronous mode
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # Set up the TM in synchronous mode
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)

    # Set a seed so behaviour can be repeated if necessary
    traffic_manager.set_random_device_seed(0)
    # random.seed(0)

    # We will aslo set up the spectator so we can see what we do
    # spectator = world.get_spectator()
    return client


def spawn_vehicles(carla_client: Client, blueprints: List[ActorBlueprint]):
    world = carla_client.get_world()
    spawn_points = world.get_map().get_spawn_points()

    # Set a max number of vehicles and prepare a list for those we spawn
    max_vehicles = 50
    max_vehicles = min([max_vehicles, len(spawn_points)])
    vehicles: List[Actor] = []
    traffic_manager = carla_client.get_trafficmanager()

    # Take a random sample of the spawn points and spawn some vehicles
    for i, spawn_point in enumerate(random.sample(spawn_points, max_vehicles)):
        temp = world.try_spawn_actor(random.choice(blueprints), spawn_point)
        if temp is not None and type(temp) is Vehicle:
            temp.set_autopilot(True, traffic_manager.get_port())
            traffic_manager.ignore_lights_percentage(temp, random.randint(0, 50))
            vehicles.append(temp)


def get_standard_blueprints(world) -> List[ActorBlueprint]:
    models = [
        "dodge",
        "audi",
        "model3",
        "mini",
        "mustang",
        "lincoln",
        "prius",
        "nissan",
        "crown",
        "impala",
    ]
    blueprints = []
    for vehicle in world.get_blueprint_library().filter("*vehicle*"):
        if any(model in vehicle.id for model in models):
            blueprints.append(vehicle)
    return blueprints


def center_camera(spectator: Actor, pedestrian: Actor, rot_offset=0):
    # Rotate the camera to face the pedestrian and apply an offset
    trans = pedestrian.get_transform()
    offset_radians = 2 * math.pi * rot_offset / 360
    x = math.cos(offset_radians) * -2
    y = math.sin(offset_radians) * 2
    trans.location.x += x
    trans.location.y += y
    trans.location.z = 2
    trans.rotation.pitch = -16
    trans.rotation.roll = 0
    trans.rotation.yaw = -rot_offset
    spectator.set_transform(trans)
    return trans


def spawn_pedestrians(carla_client: Client):
    world = carla_client.get_world()
    spectator = world.get_spectator()
    pedestrian_bp = random.choice(
        world.get_blueprint_library().filter("*walker.pedestrian*")
    )

    transform = Transform(Location(x=-134, y=78.1, z=1.18), Rotation())
    pedestrian = world.try_spawn_actor(pedestrian_bp, transform)

    # Spawn an RGB camera
    camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    camera = cast(Sensor, world.spawn_actor(camera_bp, transform))

    # Create a queue to store and retrieve the sensor data
    image_queue: queue.Queue[int] = queue.Queue()
    camera.listen(image_queue.put)

    world.tick()
    image_queue.get()
    # We must call image_queue.get() each time we call world.tick() to
    # ensure the timestep and sensor data stay synchronised

    # Now we will rotate the camera to face the pedestrian
    camera.set_transform(center_camera(pedestrian, spectator))
    # Move the spectator to see the result
    spectator.set_transform(camera.get_transform())

    # Set up the AI controller for the pedestrian.... see below
    controller_bp = world.get_blueprint_library().find("controller.ai.walker")
    controller = cast(
        WalkerAIController,
        world.spawn_actor(controller_bp, pedestrian.get_transform(), pedestrian),
    )

    # Start the controller and give it a random location to move to
    controller.start()
    controller.go_to_location(world.get_random_location_from_navigation())
    # Move the world a few frames to let the pedestrian spawn
    for _ in range(0, 5):
        world.tick()
        image_queue.get()


def main():
    print("some")
    carla_client = get_carla_client()
    world = carla_client.get_world()
    standard_blueprints = get_standard_blueprints(world)
    # Draw the spawn point locations as numbers in the map for debugging
    # for i, spawn_point in enumerate(spawn_points):
    #     world.debug.draw_string(spawn_point.location, str(i), life_time=10)
    spawn_vehicles(carla_client, standard_blueprints)
    # spawn_pedestrians(carla_client)

    # In synchronous mode, we need to run the simulation to fly the spectator
    print("Vehicles spawned")
    while True:
        world.tick()


if __name__ == "__main__":
    main()
