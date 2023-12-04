import sys
from typing import List, Protocol

import carla


class CarlaTask(Protocol):
    def __call__(
        self, world: carla.World, actors: List[carla.Actor]
    ) -> List[carla.Actor]:
        ...


def game_loop(
    world: carla.World,
    tasks: List[CarlaTask],
    initial_actors: List[carla.Actor],
):
    actors = [*initial_actors]
    while True:
        try:  # in case of a crash, try to recover and continue
            for task in tasks:
                actors = task(world, actors)
            world.tick()
        except KeyboardInterrupt:
            for actor in initial_actors:
                actor.destroy()
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
