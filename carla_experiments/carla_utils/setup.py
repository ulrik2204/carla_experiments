from typing import Any, Mapping, TypeVar

import carla

TActors = TypeVar("TActors", bound=Mapping[str, Any])

# This does not work, for some reason CARLA does not like having the game loop in a separate function
# def game_loop(
#     world: carla.World,
#     tasks: List[CarlaTask[TActors]],
#     actors: TActors,
# ):
#     # actors = cast(TActors, {**initial_actors})
#     while True:
#         try:  # in case of a crash, try to recover and continue
#             for task in tasks:
#                 task(world, actors)  # type: ignore
#             time.sleep(0.01)
#             world.tick()
#         except (KeyboardInterrupt, Exception):
#             print("Exiting...")
#             for actor in actors:  # TODO: .values()
#                 actor.destroy()  # type: ignore
#             sys.exit()


def initialize_carla():
    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True  # Enables synchronous mode
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    return client, world
