import random
import threading
import time
from typing import Callable, List, Optional, Tuple, TypeVar, cast

import carla

from carla_experiments.carla_utils.types_carla_utils import SensorBlueprint

TSensorData = TypeVar("TSensorData", bound=carla.SensorData)


def spawn_sensor(
    world: carla.World,
    blueprint: SensorBlueprint[TSensorData],
    location: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    attach_to: Optional[carla.Actor] = None,
    attachment_type: carla.AttachmentType = carla.AttachmentType.Rigid,
    modify_blueprint_fn: Optional[
        Callable[[carla.ActorBlueprint], carla.ActorBlueprint]
    ] = None,
    on_measurement_received: Callable[[TSensorData], None] = lambda _: None,
) -> carla.Sensor:
    sensor_blueprint = world.get_blueprint_library().find(blueprint.name)
    sensor_blueprint = (
        modify_blueprint_fn(sensor_blueprint)
        if modify_blueprint_fn
        else sensor_blueprint
    )
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

    return sensor_object


def control_vehicle(
    vehicle: carla.Vehicle, throttle: float, steer: float, brake: float
) -> None:
    control = carla.VehicleControl(throttle, steer, brake)
    vehicle.apply_control(control)


def spawn_vehicle(
    world: carla.World,
    blueprint: Optional[str] = None,
    autopilot: bool = False,
    spawn_point: Optional[carla.Transform] = None,
    set_attributes: Optional[
        Callable[[carla.ActorBlueprint], carla.ActorBlueprint]
    ] = None,
    tick: bool = True,
) -> carla.Vehicle:
    blueprints = world.get_blueprint_library().filter(
        "vehicle.*" if blueprint is None else blueprint
    )
    vehicle_bp = random.choice(blueprints)
    # vehicle_bp.set_attribute("role_name", "ego")
    # vehicle_color = vehicle_bp.get_attribute("color").recommended_values[0]
    # vehicle_bp.set_attribute("color", vehicle_color)
    vehicle_bp = (
        set_attributes(vehicle_bp) if set_attributes is not None else vehicle_bp
    )

    spawn_points = world.get_map().get_spawn_points()
    number_of_spawn_points = len(spawn_points)

    if number_of_spawn_points > 0:
        transform = spawn_point if spawn_point else random.choice(spawn_points)

        vehicle = cast(carla.Vehicle, world.spawn_actor(vehicle_bp, transform))
    else:
        raise Exception("Could not find any spawn points")

    if autopilot:
        # Sleep before setting autopilot is important because of timing issues.
        time.sleep(0.1)
        if tick:
            world.tick()
            time.sleep(0.1)
        vehicle.set_autopilot(True)
        print("Autopilot set")

    return vehicle


def spawn_ego_vehicle_old(
    world: carla.World,
    blueprint: str = "vehicle.tesla.model3",
    autopilot: bool = False,
    choose_spawn_point: Optional[
        Callable[[List[carla.Transform]], carla.Transform]
    ] = None,
) -> carla.Vehicle:
    ego_bp = world.get_blueprint_library().find(blueprint)
    ego_bp.set_attribute("role_name", "ego")
    ego_color = ego_bp.get_attribute("color").recommended_values[0]
    ego_bp.set_attribute("color", ego_color)

    spawn_points = world.get_map().get_spawn_points()
    number_of_spawn_points = len(spawn_points)

    if number_of_spawn_points == 0:
        raise Exception("Could not find any spawn points")

    ego_transform = (
        choose_spawn_point(spawn_points)
        if choose_spawn_point
        else random.choice(spawn_points)
    )
    ego_vehicle = cast(carla.Vehicle, world.spawn_actor(ego_bp, ego_transform))

    if autopilot:
        # Sleep before setting autopilot is important because of timing issues.
        time.sleep(1)
        world.tick()
        time.sleep(3)
        ego_vehicle.set_autopilot(True)
        print("Autopilot set")

    return ego_vehicle


def spawn_ego_vehicle(
    world: carla.World,
    blueprint: str = "vehicle.tesla.model3",
    autopilot: bool = True,
    spawn_point: Optional[carla.Transform] = None,
    choose_color: Optional[Callable[[List[str]], str]] = None,
) -> carla.Vehicle:
    def set_attributes(blueprint: carla.ActorBlueprint) -> carla.ActorBlueprint:
        blueprint.set_attribute("role_name", "hero")
        if not blueprint.has_attribute("color"):
            return blueprint
        recommended_colors = blueprint.get_attribute("color").recommended_values
        vehicle_color = (
            choose_color(recommended_colors) if choose_color else recommended_colors[0]
        )
        blueprint.set_attribute("color", vehicle_color)
        return blueprint

    return spawn_vehicle(
        world,
        blueprint,
        autopilot,
        spawn_point=spawn_point,
        set_attributes=set_attributes,
    )


def _try_spawn_vehicle_bot(
    world: carla.World, possible_spawn_points: List[carla.Transform], tries: int = 5
) -> carla.Vehicle:
    if tries == 0:
        raise RuntimeError("Could not spawn vehicle bot")
    spawn_point = random.choice(possible_spawn_points)
    try:
        vehicle = spawn_vehicle(
            world, spawn_point=spawn_point, autopilot=True, tick=False
        )
        print("vehicle spawned")
        return vehicle
    except RuntimeError as e:
        if str(e) == "Spawn failed because of collision at spawn position":
            print("spawn collision while spawning vehicle, retrying")
            return _try_spawn_vehicle_bot(world, possible_spawn_points, tries=tries - 1)
        raise e


def spawn_vehicle_bots(
    world: carla.World,
    number_of_vehicles: int,
) -> List[carla.Vehicle]:
    vehicles = []
    spawn_points = world.get_map().get_spawn_points()
    for _ in range(number_of_vehicles):
        vehicle = _try_spawn_vehicle_bot(world, spawn_points)
        vehicles.append(vehicle)
    time.sleep(0.1)
    world.tick()
    return vehicles


def _walk_towards_goal(
    world: carla.World,
    walker: carla.Walker,
    controller: carla.WalkerAIController,
    walker_path: List[carla.Location],
    walk_randomly_afterwards: bool = False,
):
    for goal in walker_path:
        try:
            controller.go_to_location(goal)
            controller.set_max_speed(
                1 + random.random()
            )  # random speed between 1 and 2

            # Wait for the walker to reach the destination (with some threshold)
            while walker.get_transform().location.distance(goal) > 1.0:
                time.sleep(1)

            time.sleep(random.randint(1, 5))
        # If the walker is destroyed, stop the loop
        except RuntimeError:
            break
    if walk_randomly_afterwards:
        _random_walking_behaviour(world, walker, controller)


def _random_walking_behaviour(
    world: carla.World, walker: carla.Walker, controller: carla.WalkerAIController
):
    while True:
        # Choose a random destination
        try:
            destination = world.get_random_location_from_navigation()
            controller.go_to_location(destination)
            controller.set_max_speed(
                1 + random.random()
            )  # random speed between 1 and 2

            # Wait for the walker to reach the destination (with some threshold)
            while walker.get_transform().location.distance(destination) > 1.0:
                time.sleep(1)

            # Optional: Wait for some time before moving to the next destination
            time.sleep(random.randint(1, 5))
        # If the walker is destroyed, stop the loop
        except RuntimeError:
            break


def spawn_walker(
    world: carla.World,
    blueprint: Optional[str] = None,
    spawn_point: Optional[carla.Transform] = None,
    walker_path: Optional[List[carla.Location]] = None,
    walk_randomly_afterwards: bool = True,
    tick: bool = True,
) -> Tuple[carla.Walker, carla.WalkerAIController]:
    blueprint_library = world.get_blueprint_library()
    walker_bp = (
        random.choice(blueprint_library.filter("walker.pedestrian.*"))
        if blueprint is None
        else blueprint_library.find(blueprint)
    )
    if walker_bp.has_attribute("is_invincible"):
        walker_bp.set_attribute("is_invincible", "false")
    if walker_bp.has_attribute("speed"):
        _, walk, run, *_ = walker_bp.get_attribute("speed").recommended_values
        speed = random.choices([walk, run], weights=[0.8, 0.2], k=1)[0]
        walker_bp.set_attribute("speed", speed)

    used_spawn_point = spawn_point or random.choice(world.get_map().get_spawn_points())
    walker = cast(carla.Walker, world.spawn_actor(walker_bp, used_spawn_point))
    print("used_spawn_point", _format_location(used_spawn_point.location))

    walker_controller_bp = world.get_blueprint_library().find("controller.ai.walker")
    walker_controller = cast(
        carla.WalkerAIController,
        world.spawn_actor(
            walker_controller_bp,
            # carla.Transform(location=carla.Location(), rotation=carla.Rotation()),
            used_spawn_point,
            attach_to=walker,
        ),
    )
    if tick:
        world.tick()
    walker_controller.start()
    if walker_path is not None:
        path_thread = threading.Thread(
            target=_walk_towards_goal,
            args=(
                world,
                walker,
                walker_controller,
                walker_path,
                walk_randomly_afterwards,
            ),
            daemon=True,
        )
        path_thread.start()
        # TODO: Handle stopping the thread when the walker is destroyed

    elif walk_randomly_afterwards:
        # destination = world.get_random_location_from_navigation()
        # walker_controller.go_to_location(destination)
        walk_random_thread = threading.Thread(
            target=_random_walking_behaviour,
            args=(world, walker, walker_controller),
            daemon=True,
        )
        walk_random_thread.start()

    return walker, walker_controller


def _format_location(location: carla.Location):
    return f"Location: ({location.x}, {location.y}, {location.z})"


def spawn_walker_bots(
    world: carla.World,
    number_of_pedestrians: int,
) -> List[Tuple[carla.Walker, carla.WalkerAIController]]:
    walkers = []
    for _ in range(number_of_pedestrians):
        walker, controller = _try_spawn_walker_bot(world)
        walkers.append((walker, controller))

    world.get_spectator().set_transform(walkers[0][0].get_transform())
    for walker, controller in walkers:
        print("walker location: ", _format_location(walker.get_location()))

    return walkers


def _try_spawn_walker_bot(
    world: carla.World, tries: int = 10
) -> Tuple[carla.Walker, carla.WalkerAIController]:
    if tries == 0:
        raise RuntimeError("Could not spawn walker bot, even after retries.")
    try:
        random_location = world.get_random_location_from_navigation()
        print("random_loc", _format_location(random_location))
        spawn_point = carla.Transform(
            location=random_location, rotation=carla.Rotation()
        )
        walker, controller = spawn_walker(world, spawn_point=spawn_point, tick=True)
        print("walker spawned")
        return walker, controller
    except RuntimeError as e:
        if str(e) == "Spawn failed because of collision at spawn position":
            print("spawn collision while spawning walker, retrying")
            return _try_spawn_walker_bot(world, tries=tries - 1)
        raise e
