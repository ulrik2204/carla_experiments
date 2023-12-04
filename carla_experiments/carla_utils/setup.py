import sys
import time
from typing import Any, List, Mapping, TypeVar, cast

import carla

from carla_experiments.carla_utils.types_carla_utils import CarlaTask

# TActors = TypeVar("TActors", bound=Dict[str, Union[carla.Actor, carla.Vehicle]])
TActors = TypeVar("TActors", bound=Mapping[str, Any])


def game_loop(
    world: carla.World,
    tasks: List[CarlaTask[TActors]],
    initial_actors: TActors,
):
    actors = cast(TActors, {**initial_actors})
    while True:
        try:  # in case of a crash, try to recover and continue
            for task in tasks:
                actors = task(world, actors)
            time.sleep(0.05)
            world.tick()
        except (KeyboardInterrupt, Exception):
            print("Exiting...")
            for actor in initial_actors.values():
                actor.destroy()  # type: ignore
            sys.exit()


def initialize_carla():
    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True  # Enables synchronous mode
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    return client, world
